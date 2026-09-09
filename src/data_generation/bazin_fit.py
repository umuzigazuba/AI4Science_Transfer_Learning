from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from feature_extraction.basic.features import FILTERS
from feature_extraction.naming import rename_column, rename_frame

# --------------------------------------------------------------------------- CC-3 flags
#
# Degeneracy bits for ``{band}_bz_flags`` (int16). See docs/backlog.md#cc-3.
#
# These must be produced *here*, at fit time, and cannot be recovered from a finished
# feature table: unlike Potica's, most of Bazin's bounds are derived per (object, band)
# from the data itself -- ``A_bound``/``C_bound`` from the flux scatter, the timescale
# ceilings from the light-curve span. Only ``tr``'s and ``tf``'s lower bounds are
# constants, which is why the audit could measure the tr-pinned rate and nothing else.
#
# A set bit does not mean "the fit failed" -- it means the number is not trustworthy as
# a measurement of light-curve shape. `bz_{b}_ok` still reports convergence.
BZ_FLAG_A_NEG = 1 << 0        # fitted A < 0: t0 marks a MINIMUM, not a peak
BZ_FLAG_A_AT_BOUND = 1 << 1   # |A| pinned at +/- max(5*std(y), 10*median(e))
BZ_FLAG_C_AT_BOUND = 1 << 2   # baseline pinned at its bound
BZ_FLAG_T0_AT_BOUND = 1 << 3  # peak epoch pinned at the padded time bound
BZ_FLAG_TR_AT_LO = 1 << 4     # rise scale at its 0.05 d floor -- inflates fall/rise
BZ_FLAG_TR_AT_HI = 1 << 5     # rise scale at its span-derived ceiling
BZ_FLAG_TF_AT_LO = 1 << 6     # fall scale at its 0.10 d floor
BZ_FLAG_TF_AT_HI = 1 << 7     # fall scale at its span-derived ceiling
BZ_FLAG_TR_GT_SPAN = 1 << 8   # rise scale longer than the light curve itself
BZ_FLAG_TF_GT_SPAN = 1 << 9   # fall scale longer than the light curve itself
BZ_FLAG_CLIPPED = 1 << 10     # flux was clipped to median + 20*std before fitting
BZ_FLAG_FEW_POINTS = 1 << 11  # fitted with < 8 epochs (B11: the event-window refit)

# Every bit describing an unconstrained timescale.
BZ_FLAGS_TIMESCALE = (
    BZ_FLAG_TR_AT_LO | BZ_FLAG_TR_AT_HI | BZ_FLAG_TF_AT_LO | BZ_FLAG_TF_AT_HI
    | BZ_FLAG_TR_GT_SPAN | BZ_FLAG_TF_GT_SPAN
)

# `bz_tfall_over_trise` is masked on ANY of the six, numerator and denominator alike.
#
# Masking only on the denominator (rise) bits was tried, on the reasoning that CC-3's
# documented pathology is a pinned *rise* sending the ratio to 1e4-1e14, and it is much
# cheaper: 21.5 % of `extra` objects masked instead of 59.5 %. The `truncated_pre_peak`
# fixture refutes it. Cutting the light curve before the peak leaves the *fall* scale
# unconstrained, it runs to its ceiling, and the ratio reaches 45.6 against a typical 4.3
# -- a tenfold inflation that the denominator-only mask waves through into the column
# whose whole purpose is to be trustworthy. An unconstrained numerator inflates a ratio
# just as effectively as a pinned denominator.
#
# 59.5 % is therefore the honest cost, and it is a statement about the data rather than
# about the mask: for three objects in five, at least one contributing band-fit has a
# timescale the light curve cannot constrain.
BZ_FLAGS_RATIO_UNTRUSTED = BZ_FLAGS_TIMESCALE

# Why no fit exists, for ``{band}_bz_reason`` (int8). Distinct from the flags above:
# these describe the *absence* of a fit, the flags describe a suspect one. `ok == 0`
# alone collapsed all four failure causes into a single value.
BZ_REASON_OK = 0
BZ_REASON_TOO_FEW_POINTS = 1   # fewer usable epochs in the band than `min_points`
BZ_REASON_BAD_SPAN = 2         # zero or non-finite time span
BZ_REASON_MAXFEV = 3           # curve_fit exhausted maxfev without converging
BZ_REASON_OTHER_EXCEPTION = 4  # anything else raised by the fitter
BZ_REASON_NOT_ATTEMPTED = 5    # the caller declined to fit this frame at all

_BOUND_RTOL = 1e-6

# --------------------------------------------------------- point budget (B11)
#
# The model has five free parameters, so `dof = n - 5` and **n = 6 is the smallest count
# at which the fit has a degree of freedom at all** -- below it `chi2r` is 0/0 and the
# `max(1, ...)` guard would quietly report a number that means nothing.
#
# The full light curve uses 8, which buys margin. The event window cannot afford it:
# median in-window usable points are 3 (u), 4 (g), 5 (y) against 8-9 in r/i/z, so the
# 8-point gate refused 89-92 % of u/g band-fits and the whole u/g/y half of the 66-column
# `evtbz` block was NaN for most objects.
#
# Relaxing a gate is only safe if what it rejects is not itself class-informative -- the
# rule the potica sigma/tau reversal produced. Measured on 3000 `extra` objects (TDE base
# rate 6.87 %), P(TDE | passes) / P(TDE | refused) is 0.38x in g, 0.47x in y, 0.73-0.79x
# in u/i/z and 1.36x only in r. In five bands of six the refused objects are TDE-
# *enriched*: the gate was discarding the class of interest, which is the opposite of
# potica's ceilings, where "cannot be fit" was a 39x TDE depletion and therefore a
# feature. Dropping to 6 admits ~500 more g-band fits (35 TDEs) and ~530 more in y (47).
#
# `BZ_FLAG_FEW_POINTS` marks every fit made below the full-LC bar, so the relaxation is
# visible per-row rather than inferred from `bz_{b}_n`.
BZ_MIN_POINTS = 8        # full light curve
BZ_MIN_POINTS_WINDOW = 6 # event-window refit: dof >= 1

# --------------------------------------------------------------- amplitude bound (CC-3)
#
# `A_bound` scales with the observed **amplitude**, not with `std(y)`.
#
# The old rule was `max(5*std(y), 10*median(e))`, and it pinned `A` for 19-50 % of
# converged band-fits including textbook transients. Bazin's `A` is not the peak height --
# the shape term maxes at `s_max` in [0.5, 1), so `A = peak / s_max`, i.e. 1.6-1.8x the
# peak -- while `std(y)` is diluted by every quiet baseline epoch. The two therefore drift
# apart as the survey baseline grows, and at LSST's ~2200 d baseline the bound came out
# roughly a third of what the fit needed. Full derivation and the overplotted fits:
# `notebooks/03_experiments/37_bazin_amplitude_bound.ipynb`.
_A_BOUND_MULT = 5.0
_A_BOUND_WINDOW = 3


def _robust_amplitude(y: np.ndarray, window: int = _A_BOUND_WINDOW) -> float:
    """Amplitude above baseline, measured so that isolated outliers do not count.

    A rolling median requires **adjacent** elevated epochs, which is the only property
    that separates a real transient from a single noisy point: a bare `max` reads 9.13 off
    the pure-noise `flat_plus_spike` fixture (true amplitude 0), and so does an SNR cut,
    because a 10-sigma *error* outlier has high SNR by construction. The 3-point window
    reads 1.33 there while still recovering 13.63 of a true 15 on `clean_sn`.

    The window stays at 3 deliberately: the median r-band cadence is 21.5 d, so a fast
    transient may have only two or three epochs above baseline at all, and a wider window
    would start rejecting real events.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0 or not np.isfinite(y).any():
        return np.nan
    if y.size < window:
        smoothed = y
    else:
        smoothed = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    return float(np.nanmax(smoothed) - np.nanmedian(y))


def _at_bound(value: float, lo: float, hi: float) -> tuple[bool, bool]:
    """Whether ``value`` sits at the low / high end of ``[lo, hi]``.

    Tolerance scales with the magnitude of the bounds, so a 0.05 d floor and a 400 d
    ceiling are both judged sensibly.
    """
    if not np.isfinite(value):
        return False, False
    tol = _BOUND_RTOL * max(abs(lo), abs(hi), 1.0)
    return bool(value <= lo + tol), bool(value >= hi - tol)


def _to_standard_lc(lc: pd.DataFrame) -> pd.DataFrame:
    df = lc.rename(
        columns={"Time (MJD)": "mjd", "Flux": "flux", "Flux_err": "flux_err", "Filter": "filt"}
    ).copy()
    df["object_id"] = df["object_id"].astype(str)
    df["filt"] = df["filt"].astype(str)
    return df.sort_values(["object_id", "mjd"])

def bazin_col_legacy_to_prefixed(col: str) -> str:
    m = re.match(r"^([ugrizy])_bz_(.+)$", col)
    if m:
        return f"bz_{m.group(1)}_{m.group(2)}"
    return col

def bazin_col_prefixed_to_legacy(col: str) -> str:
    m = re.match(r"^bz_([ugrizy])_(.+)$", col)
    if m:
        return f"{m.group(1)}_bz_{m.group(2)}"
    return col

def _apply_bz_prefix(df: pd.DataFrame, *, cc1_names: bool = True) -> pd.DataFrame:
    """Legacy ``bz_*`` prefix, then -- unless ``cc1_names`` is off -- the CC-1 grammar.

    ``evtWin`` refits this family inside the event window and needs the legacy stem to
    re-stem (``bz_`` -> ``evtbz_``) before its own prefix goes on; the CC-1 rename then
    happens once, at that family's boundary, where ``gev_evtbz_*`` becomes ``bazinEvt_*``.
    Renaming here as well would leave the nested block under the wrong family token.
    """
    rename = {}
    for c in df.columns:
        if c == "object_id":
            continue
        rename[c] = bazin_col_legacy_to_prefixed(c)
    out = df.rename(columns=rename)
    return rename_frame(out) if cc1_names else out


def bazin(t, A, t0, tr, tf, C):
    """``A * exp(-(t-t0)/tf) / (1 + exp(-(t-t0)/tr)) + C``, evaluated in log space.

    B13. The previous implementation clipped the two exponents **independently** to
    +/-60 and then divided::

        a = clip(-x/tf, -60, 60);  b = clip(-x/tr, -60, 60)
        return A * exp(a) / (1 + exp(b)) + C

    Far before the peak both saturate at +60, so the ratio collapses to
    ``e^60 / (1 + e^60) ~ 1`` and the model returns **A + C** -- a plateau at full
    amplitude, precisely where the transient has not happened yet. The correct limit is
    ``C``.

    ⚠️ **Why this was invisible, and why it still mattered.** The clip fires for
    ``x < -60*tf``. At the *converged* parameters (median fitted ``tf`` = 102 d in r) that
    needs a light curve reaching 6137 d before the peak, so on `extra` it fires for
    **0.0 %** of final evaluations. But `curve_fit` minimises against this same function,
    and at small ``tf`` the threshold collapses: at ``tf = 5 d`` the clip fires for
    **95.7 %** of light curves, and at the 0.10 d floor for **99.6 %**. So the bug never
    corrupted a reported value -- it corrupted the objective surface the optimiser walks
    across the whole small-``tf`` region, making fast-decay solutions look like a raised
    plateau instead of a spike.

    The stable form uses ``logaddexp(0, b) == log(1 + exp(b))``, which is exact at both
    ends, so no clipping of the exponents is needed.

    The shape term is instead bounded above by **1**, and that bound is doing real work
    only outside the model's domain. For ``tr < tf`` -- a rise faster than the fall, which
    is what a transient is, and 96-99 % of converged fits -- the shape peaks at
    ``s_max`` in [0.5, 1), so the cap never binds and changes nothing. For ``tr > tf`` the
    ratio *diverges* as ``t -> -inf`` (the exponent is ``|x|*(1/tf - 1/tr) > 0``), which
    would mean the source was arbitrarily bright in the past; left unbounded it overflows
    `curve_fit`'s residual sum. Capping at 1 keeps ``A`` interpretable as the amplitude
    scale and keeps the objective finite, so the optimiser can walk out of that region
    instead of hitting an overflow there.

    This is **not** a return of the B13 plateau: that one sat at ``A + C`` in the
    *physical* regime, this one engages only where the rise is slower than the fall.
    """
    tr = np.clip(tr, 1e-3, None)
    tf = np.clip(tf, 1e-3, None)
    x = t - t0

    log_shape = (-x / tf) - np.logaddexp(0.0, -x / tr)
    return A * np.exp(np.clip(log_shape, -700.0, 0.0)) + C


def _initial_guess(t, y):
    y_med = float(np.median(y))
    i_peak = int(np.argmax(y))
    t0 = float(t[i_peak])
    A = float(np.max(y) - y_med)
    if not np.isfinite(A) or abs(A) < 1e-6:
        A = float(np.std(y)) if np.isfinite(np.std(y)) and np.std(y) > 0 else 1.0
    C = y_med
    span = float(np.max(t) - np.min(t))
    if not np.isfinite(span) or span <= 0:
        span = 10.0
    tr = max(0.5, 0.05 * span)
    tf = max(1.0, 0.20 * span)
    return A, t0, tr, tf, C


def _fit_bazin_one_band(t, y, e, min_points: int = BZ_MIN_POINTS):
    out = dict(
        A=np.nan, t0=np.nan, tr=np.nan, tf=np.nan, C=np.nan, chi2_red=np.nan,
        n_pts=int(len(t)), success=0, flags=0, reason=BZ_REASON_OTHER_EXCEPTION,
    )
    if len(t) < min_points:
        out["reason"] = BZ_REASON_TOO_FEW_POINTS
        return out
    flags_pre = BZ_FLAG_FEW_POINTS if len(t) < BZ_MIN_POINTS else 0
    span = float(np.max(t) - np.min(t))
    if not np.isfinite(span) or span <= 0:
        out["reason"] = BZ_REASON_BAD_SPAN
        return out

    order = np.argsort(t)
    t = t[order].astype(float)
    y = y[order].astype(float)
    e = e[order].astype(float)
    e = np.clip(e, 1e-6, np.percentile(e, 99.5))

    p0 = _initial_guess(t, y)

    A_bound = max(_A_BOUND_MULT * _robust_amplitude(y), 10.0 * np.nanmedian(e))
    if not np.isfinite(A_bound) or A_bound <= 0:
        A_bound = 1e3

    tmin, tmax = float(np.min(t)), float(np.max(t))
    tr_min, tr_max = 0.05, max(2.0, 2.0 * span)
    tf_min, tf_max = 0.10, max(5.0, 5.0 * span)

    # The baseline bound stays tied to the flux *scatter*, and is no longer the same
    # number as `A_bound`. Previously both were `max(5*std, 10*median(e))`, so any change
    # to the amplitude rule would have silently dragged the baseline bound with it -- two
    # unrelated quantities sharing one expression.
    C_bound = max(5.0 * np.nanstd(y), 10.0 * np.nanmedian(e))
    if not np.isfinite(C_bound) or C_bound <= 0:
        C_bound = 1e3

    bounds_lo = (-A_bound, tmin - 0.2 * span, tr_min, tf_min, -C_bound)
    bounds_hi = (A_bound, tmax + 0.2 * span, tr_max, tf_max, C_bound)

    flags = flags_pre
    try:
        y_clip = np.nanmedian(y) + 20.0 * np.nanstd(y) if np.isfinite(np.nanstd(y)) and np.nanstd(y) > 0 else None
        if y_clip is not None and np.isfinite(y_clip):
            # Record that the fitter modified its own input before doing so; otherwise a
            # fit against clipped data is indistinguishable from one against the real data.
            if np.any(np.abs(y) > y_clip):
                flags |= BZ_FLAG_CLIPPED
            y = np.clip(y, -y_clip, y_clip)

        popt, _ = curve_fit(
            bazin,
            t,
            y,
            p0=p0,
            sigma=e,
            absolute_sigma=True,
            bounds=(bounds_lo, bounds_hi),
            maxfev=20000,
        )
        A, t0, tr, tf, C = popt
        yhat = bazin(t, *popt)
        resid = (y - yhat) / e
        chi2 = float(np.sum(resid**2))
        dof = max(1, len(t) - len(popt))

        if A < 0:
            flags |= BZ_FLAG_A_NEG
        if any(_at_bound(A, bounds_lo[0], bounds_hi[0])):
            flags |= BZ_FLAG_A_AT_BOUND
        if any(_at_bound(C, bounds_lo[4], bounds_hi[4])):
            flags |= BZ_FLAG_C_AT_BOUND
        if any(_at_bound(t0, bounds_lo[1], bounds_hi[1])):
            flags |= BZ_FLAG_T0_AT_BOUND
        tr_lo, tr_hi = _at_bound(tr, tr_min, tr_max)
        flags |= (BZ_FLAG_TR_AT_LO if tr_lo else 0) | (BZ_FLAG_TR_AT_HI if tr_hi else 0)
        tf_lo, tf_hi = _at_bound(tf, tf_min, tf_max)
        flags |= (BZ_FLAG_TF_AT_LO if tf_lo else 0) | (BZ_FLAG_TF_AT_HI if tf_hi else 0)
        if tr > span:
            flags |= BZ_FLAG_TR_GT_SPAN
        if tf > span:
            flags |= BZ_FLAG_TF_GT_SPAN

        out.update(
            A=float(A), t0=float(t0), tr=float(tr), tf=float(tf), C=float(C),
            chi2_red=float(chi2 / dof), success=1, flags=int(flags), reason=BZ_REASON_OK,
        )
        return out
    except RuntimeError:
        # curve_fit's non-convergence path, which in practice means maxfev exhausted.
        out.update(flags=int(flags), reason=BZ_REASON_MAXFEV)
        return out
    except Exception:
        out.update(flags=int(flags), reason=BZ_REASON_OTHER_EXCEPTION)
        return out


def _bazin_features_for_object(oid: str, g: pd.DataFrame, include_extended_summary: bool,
                               min_points: int = BZ_MIN_POINTS) -> dict:
    out = {"object_id": oid}

    # `nanmin` raises on an empty frame rather than returning NaN, so the guard has to
    # come first: `declined_bazin_row` deliberately passes no rows at all.
    t_all = g["mjd"].to_numpy(dtype=float)
    t_all = t_all[np.isfinite(t_all)]
    t_ref = float(t_all.min()) if t_all.size else 0.0

    # CC-2 Rule C: the mark every `bz_{b}_t0` is counted from, as an absolute MJD. It is
    # the one column this family emits that a time shift moves, and it is what makes a
    # mis-anchored `t0` findable -- the whole point of B12 was that nothing in the name
    # recorded which frame the fit was done in. Note the value depends on the frame this
    # is handed: on the event-window refit it is the first *in-window* epoch, which is
    # why `evtWin` re-uses it as its own mark rather than storing a second one.
    out["bz_qc_t_ref_mjd"] = t_ref

    for f in FILTERS:
        gf = g[g["filt"] == f]
        t = gf["mjd"].to_numpy(dtype=float) - t_ref
        y = gf["flux"].to_numpy(dtype=float)
        e = gf["flux_err"].to_numpy(dtype=float)

        m = np.isfinite(t) & np.isfinite(y) & np.isfinite(e)
        t, y, e = t[m], y[m], e[m]

        fit = _fit_bazin_one_band(t, y, e, min_points=min_points)

        out[f"{f}_bz_A"] = fit["A"]
        out[f"{f}_bz_t0"] = fit["t0"]
        out[f"{f}_bz_tr"] = fit["tr"]
        out[f"{f}_bz_tf"] = fit["tf"]
        out[f"{f}_bz_C"] = fit["C"]
        out[f"{f}_bz_chi2r"] = fit["chi2_red"]
        out[f"{f}_bz_n"] = fit["n_pts"]
        out[f"{f}_bz_ok"] = fit["success"]
        out[f"{f}_bz_flags"] = fit["flags"]
        out[f"{f}_bz_reason"] = fit["reason"]

    for a, b in [("g", "r"), ("r", "i"), ("i", "z"), ("u", "g"), ("z", "y")]:
        ta, tb = out.get(f"{a}_bz_t0", np.nan), out.get(f"{b}_bz_t0", np.nan)
        out[f"bz_dt0_{a}{b}"] = (ta - tb) if (np.isfinite(ta) and np.isfinite(tb)) else np.nan
        tfa, tfb = out.get(f"{a}_bz_tf", np.nan), out.get(f"{b}_bz_tf", np.nan)
        out[f"bz_dtf_{a}{b}"] = (tfa - tfb) if (np.isfinite(tfa) and np.isfinite(tfb)) else np.nan

    ok = [out.get(f"{f}_bz_ok", 0) for f in FILTERS]
    out["bz_nfit"] = int(np.sum(ok))

    tfs = [out.get(f"{f}_bz_tf", np.nan) for f in FILTERS if out.get(f"{f}_bz_ok", 0) == 1]
    trs = [out.get(f"{f}_bz_tr", np.nan) for f in FILTERS if out.get(f"{f}_bz_ok", 0) == 1]
    amps = [out.get(f"{f}_bz_A", np.nan) for f in FILTERS if out.get(f"{f}_bz_ok", 0) == 1]
    chi = [out.get(f"{f}_bz_chi2r", np.nan) for f in FILTERS if out.get(f"{f}_bz_ok", 0) == 1]

    out["bz_tf_med"] = float(np.nanmedian(tfs)) if tfs else np.nan
    out["bz_tr_med"] = float(np.nanmedian(trs)) if trs else np.nan
    out["bz_A_med"] = float(np.nanmedian(amps)) if amps else np.nan
    out["bz_chi2r_med"] = float(np.nanmedian(chi)) if chi else np.nan
    out["bz_tfall_over_trise"] = (
        float(out["bz_tf_med"] / out["bz_tr_med"])
        if (np.isfinite(out["bz_tf_med"]) and np.isfinite(out["bz_tr_med"]) and out["bz_tr_med"] > 0)
        else np.nan
    )

    # CC-3 "emit both": the raw ratio above keeps its value even when an unconstrained
    # timescale sends it to 1e5, and this twin is NaN whenever any contributing band-fit
    # had a timescale on a bound or longer than the light curve. Which of the two a model
    # actually prefers is left to the feature-selection experiment rather than decided here.
    contributing = [
        out.get(f"{f}_bz_flags", 0) for f in FILTERS if out.get(f"{f}_bz_ok", 0) == 1
    ]
    out["bz_tfall_over_trise_masked"] = (
        np.nan
        if any(int(fl) & BZ_FLAGS_RATIO_UNTRUSTED for fl in contributing)
        else out["bz_tfall_over_trise"]
    )

    if include_extended_summary:
        amps_abs = [abs(out.get(f"{f}_bz_A", np.nan)) for f in FILTERS if out.get(f"{f}_bz_ok", 0) == 1]
        out["bz_absA_med"] = float(np.nanmedian(amps_abs)) if amps_abs else np.nan
        out["bz_fit_frac"] = out["bz_nfit"] / 6.0

    return out


_EMPTY_STANDARD_LC = pd.DataFrame({"mjd": [], "flux": [], "flux_err": [], "filt": []})


def declined_bazin_row(
    oid: str,
    *,
    include_extended_summary: bool = False,
    reason: int = BZ_REASON_NOT_ATTEMPTED,
    prefix: bool = True,
    cc1_names: bool = True,
) -> dict:
    """The full `bz_*` column set for an object that was deliberately **not** fitted.

    Every parameter is `NaN`, every count and flag is 0, and every `reason` says why. Used
    by `evtWin` on its fallback objects (C8): when no event window was detected the window
    degenerates to the whole light curve, so a windowed refit there is not a windowed
    refit at all -- it silently reproduces the full-LC `bz_*` fit under an `evtbz_` name.

    Built by running the ordinary per-object path over an empty frame, so the key set can
    never drift from the fitted one: every band takes the too-few-points branch before any
    fitting happens, which costs nothing.
    """
    row = _bazin_features_for_object(oid, _EMPTY_STANDARD_LC, include_extended_summary)
    for f in FILTERS:
        row[f"{f}_bz_reason"] = reason
    row["bz_qc_t_ref_mjd"] = np.nan
    if prefix:
        row = {bazin_col_legacy_to_prefixed(k) if k != "object_id" else k: v
               for k, v in row.items()}
        if cc1_names:
            row = {rename_column(k): v for k, v in row.items()}
    return row


def build_bazin_features(
    lc: pd.DataFrame,
    n_jobs: int = -1,
    *,
    include_extended_summary: bool = False,
    prefix: bool = True,
    cc1_names: bool = True,
    verbose: bool = False,
    progress_every: int = 100,
    min_points: int = BZ_MIN_POINTS,
) -> pd.DataFrame:
    """Fit the Bazin model per band for every object.

    ``min_points`` is the usable-epoch bar a band must clear before a fit is attempted.
    The default suits the full light curve; the event-window refit passes
    ``BZ_MIN_POINTS_WINDOW`` (B11). Any fit made below ``BZ_MIN_POINTS`` carries
    ``BZ_FLAG_FEW_POINTS``.

    ``prefix`` selects the published names over the bare internal ones; ``cc1_names``
    then puts those published names into the CC-1 grammar. Only ``evtWin``'s nested
    refit turns it off -- see :func:`_apply_bz_prefix`.
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
                print(f"[bazin] {i}/{total}", flush=True)
            rows.append(_bazin_features_for_object(oid, g, include_extended_summary,
                                                   min_points=min_points))
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=16)(
            delayed(_bazin_features_for_object)(oid, g, include_extended_summary, min_points)
            for oid, g in groups
        )

    feat = pd.DataFrame(rows)
    for f in FILTERS:
        for c in [f"{f}_bz_n", f"{f}_bz_ok"]:
            if c in feat.columns:
                feat[c] = feat[c].fillna(0).astype(int)
        # Explicit integer dtypes, never bool: `select_dtypes(include=np.number)` drops
        # bool columns, which is how gev_has_event vanished from every pool (C8/CC-4).
        # The flags mask needs 12 bits, hence int16 rather than CC-4's nominal int8.
        if f"{f}_bz_flags" in feat.columns:
            feat[f"{f}_bz_flags"] = feat[f"{f}_bz_flags"].fillna(0).astype("int16")
        if f"{f}_bz_reason" in feat.columns:
            feat[f"{f}_bz_reason"] = feat[f"{f}_bz_reason"].fillna(BZ_REASON_OTHER_EXCEPTION).astype("int8")
    if "bz_nfit" in feat.columns:
        feat["bz_nfit"] = feat["bz_nfit"].fillna(0).astype(int)
    if prefix:
        return _apply_bz_prefix(feat, cc1_names=cc1_names)
    return feat
