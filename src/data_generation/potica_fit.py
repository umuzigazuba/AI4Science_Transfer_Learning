"""Potica light-curve fit features (Gaussian-rise + exponential-decay, jointly fitted).

This is a fast, parallel re-implementation of the fitting loop in
``notebooks_refactor/16_Potica_reproduce.ipynb``.  The model is a shared-shape
Gaussian-rise + exponential-decay light curve fit jointly across the six bands
(u, g, r, i, z, y) per object, with per-band amplitude offsets relative to the
reference band (g).

Objects are fitted in parallel (joblib), a single ``groupby("object_id")`` replaces
the notebook's per-object full-dataframe scan, and the chi-squared inner loop
computes the model one band at a time with the finite-flux mask applied once per
object rather than on every function evaluation.

Three faithful-to-the-submission quirks were **removed** in 2026-08-10, so the
values are no longer bit-for-bit identical to the original notebook (A1 and CC-3;
see ``docs/backlog/potica.md``):

1. **The offset labels were rotated by one.** The chi-squared assigns offsets to
   ``[g, r, i, u, y, z]`` as ``(0, x4, x5, x6, x7, x8)``, but the output read them
   off as ``green=x4, red=x5, i=x6, u=x7, y=x8, z=x9`` -- so ``offset_green`` held
   red's offset, and so on down the list. Every band label in the colour block was
   wrong.
2. **The tenth parameter did nothing.** ``x9`` never entered the chi-squared, so it
   had no gradient and the optimiser left it at its initial 0 for 99.7 % of
   objects. ``offset_z`` was therefore a constant, which made ``z_p == A`` and left
   all 33 z-labelled columns carrying no z-band information at all. The parameter
   vector is now 9 entries, and ``offset_z`` holds the real z-band offset.
3. **Rows were not sorted**, so the chi-squared summed residuals in input order and
   every fitted parameter shifted by ~1e-8 when the rows were permuted. They are
   sorted now, like every other extractor.

The reference band (``g``) has offset 0 by construction, so it has no
``offset_*`` column; ``green_p`` is therefore exactly ``A``.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

from feature_extraction.naming import rename_frame

# Band order used throughout, matching the notebook's ``data_dict`` insertion
# order (green, red, indigo, violet, yellow, darkslategray).  Green is the
# reference band (offset fixed to 0).
_BANDS = ["g", "r", "i", "u", "y", "z"]
_REF_BAND = "g"
# The five fitted offsets, in the order they occupy the parameter vector. This list
# IS the mapping from `params[4:]` to band, and the output columns are now named
# from it rather than by position.
_OFFSET_BANDS = [b for b in _BANDS if b != _REF_BAND]      # r, i, u, y, z
_N_PARAMS = 4 + len(_OFFSET_BANDS)                          # 9, was 10

# Column name carrying each band's offset. The reference band has none: its offset
# is 0 by construction, and emitting a constant-zero column would only add a
# feature that cannot discriminate anything (CC-4).
_OFFSET_COL = {b: f"offset_{'red' if b == 'r' else b}" for b in _OFFSET_BANDS}

# --------------------------------------------------------------------- CC-3 flags
#
# Degeneracy bits for ``pt_flags`` (int16). Unlike Bazin's, most of Potica's bounds
# are constants, so these rates can also be backfilled onto existing tables.
#
# Rates and TDE depletion below are on 3000 `extra` objects (175 TDEs) after the
# 2026-08-12 bound work; a bracketed figure is the pre-change rate. **The ceiling flags
# are among the most discriminative columns the family emits** -- see the shape-ceiling
# note further down for why that means the bounds must not be widened to make fits
# converge.
PT_FLAG_SIGMA_AT_HI = 1 << 0      # rise width at its 300 d ceiling (16.7 %), 35x depleted
                                  # in TDEs -- a TDE rise is months, so this fires on AGN
                                  # and on badly-sampled objects
PT_FLAG_SIGMA_COLLAPSED = 1 << 1  # rise width < 0.1 d, an effectively instantaneous
                                  # rise (4.6 %) [11.2 %], 2x depleted
PT_FLAG_TAU_AT_HI = 1 << 2        # decay time at its 600 d ceiling (18.2 %), 39x depleted
PT_FLAG_TAU_COLLAPSED = 1 << 3    # decay time < 0.1 d (1.0 %)
PT_FLAG_TPEAK_AT_BOUND = 1 << 4   # peak epoch at +/-100 d from the brightest epoch
                                  # (15.7 %), 5x depleted -- also class-informative, so
                                  # it is NOT a candidate for widening despite being the
                                  # last absolute constant among the bounds
PT_FLAG_A_AT_BOUND = 1 << 5       # amplitude at its bound (1.2 %) [0.0 %]
PT_FLAG_OFFSET_AT_BOUND = 1 << 6  # any band offset at its bound (11.4 %) [2.2 %], and
                                  # ~1x depleted, i.e. NOT class-informative. The tighter
                                  # per-object amplitude bound raised the rate five-fold
                                  # and bought nothing for classification; first thing to
                                  # reconsider if `_AMP_BOUND_MULT` is revisited
PT_FLAG_OFFSET_BOUNDS_EXCLUDE_ZERO = 1 << 7   # RETIRED 2026-08-12, always 0. The offset
                                              # interval is now centred on zero, so it
                                              # always contains it. It used to be
                                              # (A_initial-50, A_initial+50), which
                                              # excluded 0 whenever A_initial > 50 --
                                              # forcing every band brighter than the
                                              # reference, with the initial guess of 0
                                              # outside its own bounds (0.4 % of `extra`).
                                              # The bit is kept rather than renumbered so
                                              # existing tables stay readable.
PT_FLAG_PREPEAK_UNDERFLOW = 1 << 8            # sigma small enough that exp(-d^2/2sigma^2)
                                              # underflows, so the pre-peak epoch
                                              # columns are exactly 0
PT_FLAG_NOT_MOVED = 1 << 9        # the optimiser returned its initial guess: either it
                                  # never iterated, or it stepped and came back. Either
                                  # way `sigma`/`tau`/`A` are the seed values (2, 5,
                                  # max flux) rather than fitted quantities, and
                                  # `opt_success` still reads 1. Testing `nit == 0`
                                  # alone caught only 0.84 % of the 3.65 % affected.
PT_FLAG_CHI2_SENTINEL = 1 << 10   # the objective returned its 1e30 failure sentinel, so
                                  # no band's model could be evaluated at all
PT_FLAG_UNDERDETERMINED = 1 << 11  # fewer usable epochs than free parameters. Potica has
                                   # no minimum-points guard of its own (Bazin requires 8
                                   # per band), so it will happily fit 9 parameters to 3
                                   # epochs and report success

# CC-7: the `{b}_p_{30d_before,10d_before,20d_after,40d_after}` columns are evaluations of
# the *fitted model*, not measurements, so they exist whether or not the light curve covers
# that epoch. On a cut window they are routinely extrapolations. These two bits say when.
#
# **Flagged rather than NaN-ed**, which is the choice CC-7 leaves open and potica's own
# `sigma -> 0` decision already argued for: an extrapolation from a well-constrained fit is
# a legitimate model evaluation, and NaN-ing it would cascade through the 90 pairwise
# difference/ratio columns built on the 10d-before and 20d-after rungs. The flag preserves
# the value and labels it; a NaN would only destroy it.
PT_FLAG_EPOCH_BEFORE_DATA = 1 << 12  # the earliest evaluated epoch (t_peak - 30 d) falls
                                     # before the object's first observation
PT_FLAG_EPOCH_AFTER_DATA = 1 << 13   # the latest evaluated epoch (t_peak + 40 d) falls
                                     # after the object's last observation

_SIGMA_INIT, _TAU_INIT = 2.0, 5.0    # seed values for the two shape parameters
_TPEAK_HALF_WIDTH = 100.0

# ------------------------------------------------------- shape ceilings (2026-08-12)
#
# `sigma` and `tau` are bounded at **300 d and 600 d, and these are physics, not slack**.
# A TDE lasts months, not years, and the shipped constants sit at roughly 2-2.5x the p95
# of the TDE population, measured on `extra`/full (9917 objects, 511 TDEs):
#
#              median    p75      p95
#   sigma  TDE   17.9    36.3    122.0
#      non-TDE   13.6   149.7    300.0  <- piled up at the ceiling
#     tau  TDE   77.6   143.8    314.2
#      non-TDE   47.3   351.4    600.0  <- piled up at the ceiling
#
# **The clip is one of the most discriminative things this family produces.** Objects
# pinned at the ceiling are almost never TDEs:
#
#   sigma pinned at 300 d   13.2 % of objects   P(TDE|pinned) 0.15 % vs 5.91 %   39x depleted
#   tau   pinned at 600 d   17.3 % of objects   P(TDE|pinned) 0.06 % vs 6.22 %  107x depleted
#
# and it costs almost nothing on the other side: only ~2 of 511 TDEs are sigma-pinned,
# because just 1.0 % of TDEs exceed sigma = 200 d against 21.2 % of non-TDEs.
#
# ⚠️ **Do not relax these to make the fits converge.** That was tried on 2026-08-12
# (notebook 39): tying both ceilings to the light-curve span dropped median chi2 from
# 564.9 to 497.3 and took sigma-pinning to 0.0 %. It also destroyed a 39x class
# enrichment. The chi2 gain is earned mostly on non-TDE light curves, and fitting AGN
# noise better is not a win here -- **"cannot be fit within physical TDE bounds" is the
# feature.** Median chi2 is the wrong criterion for choosing a bound when the objective is
# classification.
#
# The one thing kept from that experiment: the ceiling is additionally capped by the
# light curve's own span, since a 600 d decay measured on a 200 d window is unconstrained
# whatever the physics says. On `extra`/full that binds for 0.1 % of objects, so the
# discriminative clip above is untouched; on the cut windows 16.1 % have a span under
# 600 d, which is where it does its work.
_SIGMA_MAX, _TAU_MAX = 300.0, 600.0
_SHAPE_MAX_FLOOR = 50.0   # keeps very short light curves from getting a degenerate box


def _shape_ceilings(prepped) -> tuple[float, float]:
    """Upper bounds for ``sigma`` and ``tau``: the physical ceiling, capped by the span.

    ``min(physical, span)`` rather than either alone -- the physical constant carries the
    TDE-timescale prior that makes pinning discriminative, and the span stops a timescale
    being reported longer than the data that constrains it.
    """
    times = [t for t, _, _ in prepped if t.size]
    span = np.nan
    if times:
        lo = min(float(np.min(t)) for t in times)
        hi = max(float(np.max(t)) for t in times)
        span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = np.inf
    return (
        max(_SHAPE_MAX_FLOOR, min(_SIGMA_MAX, span)),
        max(_SHAPE_MAX_FLOOR, min(_TAU_MAX, span)),
    )
_COLLAPSE_TOL = 0.1
_BOUND_TOL = 1e-6

# --------------------------------------------------------- amplitude bounds (2026-08-12)
#
# Two problems with the previous rule, `amp_lo, amp_hi = A_initial +/- 50`:
#
# 1. **The offsets were bounded on that same interval**, which is centred on `A_initial`
#    rather than on 0. The model evaluates `A + offset`, so an offset is a *difference*
#    from the reference band and its natural range brackets zero. When `A_initial > 50`
#    the interval excluded 0 outright: every band was forced brighter than the reference,
#    and the initial guess of 0 lay outside its own bounds. 0.4 % of `extra` (43/9917),
#    flagged by `PT_FLAG_OFFSET_BOUNDS_EXCLUDE_ZERO` but never fixed.
# 2. **+/-50 is an absolute number of flux units** in a fit whose parameters carry flux
#    units, so the bound tightens as an object gets brighter -- one of the two reasons
#    Potica is not flux-scale equivariant (the other is L-BFGS-B's absolute `gtol`, which
#    is untouched here). See docs/backlog/potica.md.
#
# The half-width now scales with a **robust** amplitude rather than with the raw maximum,
# for the same reason Bazin's `A_bound` does: `np.nanmax` of the best band is set by a
# single noisy epoch, so on the `flat_plus_spike` fixture (true amplitude 0) a bare max
# reads 9.13 while the 3-point rolling median reads 1.33.
_AMP_BOUND_MULT = 5.0
_AMP_BOUND_WINDOW = 3


def _robust_amplitude(y: np.ndarray, window: int = _AMP_BOUND_WINDOW) -> float:
    """Amplitude above baseline, measured so that isolated outliers do not count.

    A rolling median requires **adjacent** elevated epochs, which is what separates a real
    transient from one noisy point. Deliberately the same rule as
    `bazin_family.fit._robust_amplitude`; kept as a separate definition rather than
    imported so the two families stay independent, as they are in every other respect.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0 or not np.isfinite(y).any():
        return np.nan
    smoothed = (
        y if y.size < window
        else pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    )
    return float(np.nanmax(smoothed) - np.nanmedian(y))


def _amplitude_half_width(flux: np.ndarray, err: np.ndarray) -> float:
    """Half-width of the amplitude and offset intervals, in flux units.

    **This is a tightening, not a like-for-like rescaling.** At `_AMP_BOUND_MULT = 5` the
    half-width is narrower than the old +/-50 for the great majority of objects (median
    robust amplitude on `extra` is 1.77, so a median half-width near 9). That is
    deliberate and measured, not incidental -- an earlier draft claimed parity with the
    old constant and was simply wrong.

    Measured on 400 `extra` objects against the shipped bounds:

        mult   chi2 med   chi2 mean   sigma collapsed   offset at bound
        HEAD      689.6        4095             11.2 %             2.2 %
           3      554.0        4194              2.5 %            18.5 %
           5      564.9        3919              3.5 %            12.2 %
          10      591.4        5566              5.2 %             7.8 %
          15      622.3        5266              7.5 %             5.0 %

    Every scale-free setting beats the absolute bound on median chi2 and on collapsed
    rises, which is the counter-intuitive part: a *tighter* per-object box gives *better*
    fits. `minimize` with bounds dispatches to L-BFGS-B, and the wide absolute box let it
    wander into the flat far field where the gradient vanishes and it terminated at a poor
    point. Constraining it to the object's own flux scale keeps it in a well-conditioned
    region -- and is also what makes the fit flux-scale equivariant.

    5 was chosen (2026-08-12) for the best mean chi2 and near-best median. The cost is
    that offsets pin at the bound for 12.2 % of fits against 2.2 % before; those are fits
    wanting a band five amplitudes brighter than the reference, which is degenerate, so
    `PT_FLAG_OFFSET_AT_BOUND` firing is the correct outcome rather than a regression.
    """
    amp = _AMP_BOUND_MULT * _robust_amplitude(flux)
    floor = 10.0 * np.nanmedian(err) if np.isfinite(err).any() else np.nan
    half = np.nanmax([amp, floor])
    if not np.isfinite(half) or half <= 0:
        return 50.0  # the legacy constant, as a last resort on degenerate input
    return float(half)


def _near(value: float, target: float, scale: float) -> bool:
    return bool(np.isfinite(value) and abs(value - target) <= _BOUND_TOL * max(abs(scale), 1.0))

# --- Extended ("post-fit") feature layout, reproduced verbatim from the
# original submission code (archive RandomForest_AddedFluxRatios notebook). ---

# Per-band model flux at fixed offsets from peak.  Letter -> offset column.
# (g == green, r == red; the rest use their own short letter.)
_LETTER_OFFSET = dict(_OFFSET_COL)   # g is absent: the reference band's offset is 0
# Band order and the four (label, days, direction) epochs kept per band.
_PBAND_ORDER = ["g", "r", "i", "z", "y", "u"]
_PBAND_EPOCHS = [
    ("30d_before", 30, "before"),
    ("40d_after", 40, "after"),
    ("10d_before", 10, "before"),
    ("20d_after", 20, "after"),
]

# Peak-flux columns ({color}_p = A + offset_{color}).
_PEAK_COLORS = ["green", "red", "i", "u", "y", "z"]
# The colour-word column names are kept as they are so the CC-1 migration table stays
# valid; only the band each one refers to is corrected here.
_COLOR_TO_BAND = {"green": "g", "red": "r", "i": "i", "u": "u", "y": "y", "z": "z"}
# Peak-flux difference/ratio band pairs (named by first letter, e.g. g-r_p).
_PEAK_PAIRS = [
    ("green", "red"), ("green", "y"), ("green", "i"), ("green", "u"), ("green", "z"),
    ("red", "y"), ("red", "i"), ("red", "u"), ("red", "z"),
    ("y", "i"), ("y", "u"), ("y", "z"),
    ("i", "u"), ("i", "z"),
    ("u", "z"),
]
# Pairwise epoch (10d-before / 20d-after) color features.
_EPOCH_BANDS = ["u", "g", "r", "i", "z", "y"]
_EPOCHS = {"10d_before": "_p_10d_before", "20d_after": "_p_20d_after"}

_FAIL_ROW = {
    # CC-2 Rule C: the mark `t_peak` is counted from, as an absolute MJD. Listed here as
    # well as set in the fit so a failed object still emits the column.
    "qc_t_ref_mjd": np.nan,
    "opt_success": 0,
    "chi2": np.nan,
    "t_peak": np.nan,
    "sigma": np.nan,
    "tau": np.nan,
    "A": np.nan,
    **{col: np.nan for col in _OFFSET_COL.values()},
    "flags": 0,
    "nfev": 0,
    "nit": 0,
}


def _to_standard_lc(lc: pd.DataFrame) -> pd.DataFrame:
    """Rename input columns to the internal short names, and sort.

    Sorting matters here: the chi-squared sums residuals in row order, so an
    unsorted input made every fitted parameter depend on however the rows happened
    to arrive (~1e-8 per parameter -- the same minimum by a different summation
    path). The notebook did not sort, and matching it bit-for-bit was the reason to
    leave it alone; that argument expires with the re-run, and reproducibility under
    upstream reordering is worth more.
    """
    df = lc.rename(
        columns={"Time (MJD)": "mjd", "Flux": "flux", "Flux_err": "flux_err", "Filter": "filt"}
    ).copy()
    df["object_id"] = df["object_id"].astype(str)
    df["filt"] = df["filt"].astype(str)
    return df.sort_values(["object_id", "mjd"])


def _model_band(t: np.ndarray, t_peak: float, sigma: float, tau: float, amplitude: float) -> np.ndarray:
    """Single-band Potica model flux: Gaussian rise, exponential decay.

    ``amplitude`` is ``A + offset`` for the band.  Matches the notebook exactly,
    including the ``errstate(divide='ignore')`` (no clamping of sigma/tau).

    Each half is evaluated **only where it is used**. The previous version computed both
    arms over the whole array and selected with ``np.where``, which evaluates its arguments
    eagerly -- so the decay arm was computed for the pre-peak epochs too, where its
    exponent ``-(t - t_peak) / tau`` is *positive* and routinely exceeds 709. That raised
    ``RuntimeWarning: overflow encountered in exp`` on nearly every fit, produced ``inf``,
    and then discarded it.

    The warning was therefore never a symptom of a bad fit -- which is exactly why it was
    worth removing rather than silencing. A blanket ``over="ignore"`` would have hidden a
    real overflow just as well as this spurious one, and the overflow *is* structurally
    confined to the discarded half: ``tau`` is bounded to ``(0, tau_max)``, so on the kept
    side ``t >= t_peak`` the exponent is non-positive and can only underflow.

    Values are unchanged: ``np.where`` was already throwing these ``inf``s away.
    """
    t = np.asarray(t, dtype=float)
    pre_peak = t < t_peak
    out = np.empty(t.shape, dtype=float)
    with np.errstate(divide="ignore"):
        out[pre_peak] = np.exp(-((t[pre_peak] - t_peak) ** 2) / (2 * sigma ** 2))
        out[~pre_peak] = np.exp(-(t[~pre_peak] - t_peak) / tau)
    return amplitude * out


def _combined_chi2(params: np.ndarray, prepped: list) -> float:
    """Chi-squared across all bands for ``scipy.optimize.minimize``.

    ``prepped`` is the per-band list of ``(t, flux, flux_err)`` arrays, already
    masked once (finite flux/err and ``flux_err > 0``) in band order
    ``[g, r, i, u, y, z]``.  Green is the reference (offset 0); ``params[4:9]``
    are the offsets for r, i, u, y, z in that order.
    """
    t_peak, sigma, tau, A = params[0], params[1], params[2], params[3]
    offsets = (0.0, *params[4:_N_PARAMS])

    chi2_total = 0.0
    n_used = 0
    for (t, flux, flux_err), offset in zip(prepped, offsets):
        if t.size == 0:
            # The band is simply absent for this object, which constrains nothing --
            # it is not a reason to reject the parameter vector. Previously the empty
            # mask below took the 1e30 branch, so *any* object missing one of the six
            # bands got a constant objective, zero gradient, and a fit that terminated
            # on iteration zero with every parameter still at its initial guess --
            # reported as a success. That was 0.64 % of `extra`.
            continue
        model = _model_band(t, t_peak, sigma, tau, A + offset)
        mask = np.isfinite(model)
        if not np.any(mask):
            return 1e30      # the model itself is non-finite: reject these parameters
        resid = (flux[mask] - model[mask]) / flux_err[mask]
        chi2_total += np.sum(resid ** 2)
        n_used += 1

    if n_used == 0:
        return 1e30
    return chi2_total if np.isfinite(chi2_total) else 1e30


def _potica_fit_for_object(oid: str, g: pd.DataFrame) -> dict:
    """Fit the Potica model to a single object's light curve.

    Faithful transcription of the notebook fit loop, with the chi-squared data
    pre-masked once per object.
    """
    out = {"object_id": oid, **_FAIL_ROW}

    try:
        t_ref = float(np.nanmin(g["mjd"].to_numpy(dtype=float)))
        if not np.isfinite(t_ref):
            t_ref = 0.0
        out["qc_t_ref_mjd"] = t_ref

        # Raw per-band arrays (band order: g, r, i, u, y, z), and the masked
        # copies used by the chi-squared.  Raw arrays drive the initial-guess
        # selection exactly as the notebook does.
        raw = {}
        prepped = []
        for f in _BANDS:
            gf = g[g["filt"] == f]
            t = gf["mjd"].to_numpy(dtype=float) - t_ref
            fl = gf["flux"].to_numpy(dtype=float)
            fe = gf["flux_err"].to_numpy(dtype=float)
            raw[f] = (t, fl, fe)

            m = np.isfinite(fl) & np.isfinite(fe) & (fe > 0)
            prepped.append((t[m], fl[m], fe[m]))

        # --- choose best band by peak SNR for the initial guess ---
        candidates = []
        for f in _BANDS:
            t, fl, fe = raw[f]
            m = np.isfinite(fl) & np.isfinite(fe)
            if not np.any(m):
                continue
            t_v, f_v, fe_v = t[m], fl[m], fe[m]
            idx = int(np.argmax(f_v))
            peak_snr = f_v[idx] / fe_v[idx]
            candidates.append((f, f_v[idx], fe_v[idx], peak_snr, t_v[idx]))

        best_band = max(candidates, key=lambda x: x[3])[0]

        t_best, fl_best, _ = raw[best_band]
        A_initial = float(np.nanmax(fl_best))
        t_peak_initial = float(t_best[int(np.nanargmax(fl_best))])

        # 9-parameter vector: [t_peak, sigma, tau, A] + one offset per non-reference
        # band, in `_OFFSET_BANDS` order. The submission carried a tenth offset slot
        # that never entered the chi-squared; it has been removed (A1).
        _, fl_best_all, fe_best_all = raw[best_band]
        amp_half = _amplitude_half_width(fl_best_all, fe_best_all)
        amp_lo, amp_hi = A_initial - amp_half, A_initial + amp_half
        # The offsets are *differences* from the reference band, so their interval is
        # centred on 0 -- not on `A_initial`, which is what excluded zero entirely for
        # bright objects.
        off_lo, off_hi = -amp_half, amp_half

        sigma_max, tau_max = _shape_ceilings(prepped)

        initial_guess = [t_peak_initial, _SIGMA_INIT, _TAU_INIT, A_initial] + [0.0] * len(_OFFSET_BANDS)
        bounds = [
            (t_peak_initial - _TPEAK_HALF_WIDTH, t_peak_initial + _TPEAK_HALF_WIDTH),
            (0.0, sigma_max),
            (0.0, tau_max),
            (amp_lo, amp_hi),
        ] + [(off_lo, off_hi)] * len(_OFFSET_BANDS)

        result = minimize(_combined_chi2, initial_guess, args=(prepped,), bounds=bounds)

        x = result.x
        t_peak, sigma, tau, A = (float(x[i]) for i in range(4))
        offsets = {band: float(x[4 + i]) for i, band in enumerate(_OFFSET_BANDS)}

        flags = 0
        if _near(sigma, sigma_max, sigma_max):
            flags |= PT_FLAG_SIGMA_AT_HI
        if np.isfinite(sigma) and sigma < _COLLAPSE_TOL:
            flags |= PT_FLAG_SIGMA_COLLAPSED
        if _near(tau, tau_max, tau_max):
            flags |= PT_FLAG_TAU_AT_HI
        if np.isfinite(tau) and tau < _COLLAPSE_TOL:
            flags |= PT_FLAG_TAU_COLLAPSED
        if _near(t_peak, bounds[0][0], t_peak_initial) or _near(t_peak, bounds[0][1], t_peak_initial):
            flags |= PT_FLAG_TPEAK_AT_BOUND
        if _near(A, amp_lo, amp_half) or _near(A, amp_hi, amp_half):
            flags |= PT_FLAG_A_AT_BOUND
        if any(_near(v, off_lo, amp_half) or _near(v, off_hi, amp_half) for v in offsets.values()):
            flags |= PT_FLAG_OFFSET_AT_BOUND
        # Bit 7 is retained and can no longer fire: the offset interval is now centred on
        # zero by construction, so it always contains it. Kept rather than renumbered so
        # existing tables and the CC-3 bit table stay readable; it reads as "0.0 % of
        # objects" from the re-run onwards.
        if sum(len(t_b) for t_b, _, _ in prepped) < _N_PARAMS:
            flags |= PT_FLAG_UNDERDETERMINED
        shape_unmoved = (
            np.isclose(sigma, _SIGMA_INIT, rtol=1e-9, atol=1e-12)
            and np.isclose(tau, _TAU_INIT, rtol=1e-9, atol=1e-12)
        )
        if int(result.nit) == 0 or shape_unmoved:
            # `sigma` and `tau` are still their seed values, so the light-curve *shape*
            # was never fitted even where the amplitudes were. They enter the model
            # non-linearly, so on a flat or noisy curve their gradient vanishes while the
            # linear amplitude terms still move -- which is why checking the whole
            # parameter vector, or `nit == 0`, catches only a fraction of these.
            flags |= PT_FLAG_NOT_MOVED
        if np.isfinite(result.fun) and result.fun >= 1e29:
            flags |= PT_FLAG_CHI2_SENTINEL

        # CC-7: do the analytic epoch columns reach past the data? Times here are already
        # relative to `t_ref`, so the observed interval is [0, span]. `_PBAND_EPOCHS`
        # is the authority on the offsets rather than a repeated literal, so adding a rung
        # cannot silently escape the check.
        obs = np.concatenate([t_b for t_b, _, _ in prepped if t_b.size]) if prepped else np.array([])
        obs = obs[np.isfinite(obs)]
        if obs.size and np.isfinite(t_peak):
            pre = max(d for _, d, direction in _PBAND_EPOCHS if direction == "before")
            post = max(d for _, d, direction in _PBAND_EPOCHS if direction == "after")
            if t_peak - pre < float(obs.min()):
                flags |= PT_FLAG_EPOCH_BEFORE_DATA
            if t_peak + post > float(obs.max()):
                flags |= PT_FLAG_EPOCH_AFTER_DATA

        out.update(
            opt_success=int(bool(result.success)),
            chi2=float(result.fun),
            t_peak=t_peak,
            sigma=sigma,
            tau=tau,
            A=A,
            **{_OFFSET_COL[b]: offsets[b] for b in _OFFSET_BANDS},
            flags=int(flags),
            nfev=int(result.nfev),
            nit=int(result.nit),
        )
    except Exception:
        pass  # return the all-NaN/fail row already set above

    return out


def _add_extended_features(feat: pd.DataFrame) -> pd.DataFrame:
    """Add the post-fit color features (reproduces the submission's 132 columns).

    All values are deterministic functions of the fitted parameters, so they are
    computed vectorised over the whole result frame -- exactly the operations
    (and ``+1e-5`` quirks) of the original RandomForest_AddedFluxRatios notebook.
    """
    A = feat["A"]
    sigma = feat["sigma"]
    tau = feat["tau"]

    # Accumulate new columns and concat once (avoids fragmenting the frame).
    new: dict[str, pd.Series] = {}

    def _amp(band: str) -> pd.Series:
        """A + this band's offset. The reference band's offset is 0 by construction."""
        if band == _REF_BAND:
            return A
        return A + feat[_OFFSET_COL[band]]

    prepeak_underflow = pd.Series(False, index=feat.index)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # Per-band model flux at peak +/- N days.  Before peak -> Gaussian rise;
        # at/after peak -> exponential decay.  amplitude = A + offset_band.
        for b in _PBAND_ORDER:
            amp = _amp(b)
            for label, days, direction in _PBAND_EPOCHS:
                if direction == "before":
                    factor = np.exp(-(days ** 2) / (2 * sigma ** 2))
                    # A sufficiently collapsed rise underflows this to exactly 0, so
                    # every pre-peak column becomes 0.0 -- a value indistinguishable
                    # from a real faint flux. The values are kept (NaN would be
                    # median-imputed downstream, which is worse than a deterministic
                    # sentinel); the flag is what marks them.
                    prepeak_underflow |= (factor == 0.0) & np.isfinite(sigma)
                else:
                    factor = np.exp(-days / tau)
                new[f"{b}_p_{label}"] = amp * factor

        # Peak model flux per band ({color}_p = A + offset_{band}; model at dt=0).
        for color in _PEAK_COLORS:
            new[f"{color}_p"] = _amp(_COLOR_TO_BAND[color])

        # Peak-flux differences and ratios.
        for b1, b2 in _PEAK_PAIRS:
            new[f"{b1[0]}-{b2[0]}_p"] = new[f"{b1}_p"] - new[f"{b2}_p"]
        for b1, b2 in _PEAK_PAIRS:
            new[f"{b1[0]}/{b2[0]}_p"] = new[f"{b1}_p"] / (new[f"{b2}_p"] + 1e-5)

        # Pairwise epoch (10d-before / 20d-after) differences and ratios.
        for b1, b2 in itertools.combinations(_EPOCH_BANDS, 2):
            for ep_name, suffix in _EPOCHS.items():
                col1 = new[f"{b1}{suffix}"]
                col2 = new[f"{b2}{suffix}"]
                new[f"{b1}_{b2}_{ep_name}"] = col1 - col2
                new[f"{b1}/{b2}_{ep_name}"] = col1 / col2 + 1e-5

    out = pd.concat([feat, pd.DataFrame(new, index=feat.index)], axis=1)
    out["flags"] = (out["flags"].to_numpy(dtype=np.int64)
                    | (prepeak_underflow.to_numpy() * PT_FLAG_PREPEAK_UNDERFLOW)).astype("int16")
    return out


def build_potica_features(
    lc: pd.DataFrame,
    n_jobs: int = -1,
    *,
    extended: bool = True,
    prefix: bool = True,
    verbose: bool = False,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Extract Potica fit features for every object in a light-curve table.

    Parameters
    ----------
    lc : pd.DataFrame
        Light-curve table with columns: object_id, Time (MJD), Flux, Flux_err,
        Filter.
    n_jobs : int
        Number of parallel workers. -1 uses all available CPUs minus one.
        Set to 1 to run sequentially (useful for debugging / exact comparison).
    extended : bool
        If True (default), append the post-fit color features (per-band model
        flux at peak +/- N days, peak-flux differences/ratios, and pairwise
        epoch differences/ratios) reproducing the submission's full feature set.
    prefix : bool
        If True, emit the published column names -- the legacy ``pt_`` prefix, then the
        CC-1 grammar, which also retires the ``-`` and ``/`` that this family put into
        45 colour and ratio names and that break ``df.query()`` and every formula
        interface. If False, the bare internal names, which the tier-4 snapshots pin
        against the original submission code.
    verbose : bool
        If True, print progress and run sequentially.
    progress_every : int
        Print a progress line every this many objects (only when verbose=True).

    Returns
    -------
    pd.DataFrame
        One row per object with columns: object_id, status, chi2, t_peak, sigma,
        tau, A, offset_green, offset_red, offset_i, offset_u, offset_y, offset_z,
        nfev, nit.
    """
    df = _to_standard_lc(lc)
    groups = [(oid, g) for oid, g in df.groupby("object_id", sort=False)]

    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    if verbose:
        rows = []
        total = len(groups)
        for i, (oid, g) in enumerate(groups, start=1):
            if i % progress_every == 0 or i == total:
                print(f"[potica] {i}/{total}", flush=True)
            rows.append(_potica_fit_for_object(oid, g))
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=16)(
            delayed(_potica_fit_for_object)(oid, g) for oid, g in groups
        )

    feat = pd.DataFrame(rows)
    for col in ("nfev", "nit", "opt_success"):
        if col in feat.columns:
            feat[col] = feat[col].fillna(0).astype(int)
    # Explicit integer dtype, never bool or str: `select_dtypes(include=np.number)`
    # drops both, which is how the old string `status` column never reached a model.
    if "flags" in feat.columns:
        feat["flags"] = feat["flags"].fillna(0).astype("int16")

    if extended:
        feat = _add_extended_features(feat)

    if prefix:
        rename = {c: f"pt_{c}" for c in feat.columns if c != "object_id"}
        return rename_frame(feat.rename(columns=rename))
    return feat
