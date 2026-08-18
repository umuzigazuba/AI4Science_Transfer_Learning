# mallorn_ai/src/feature_extraction/event_window.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

FILTERS = list("ugrizy")


@dataclass(frozen=True)
class EventWindowParams:
    filters: Tuple[str, ...] = tuple(FILTERS)

    # smoothing (in points) for significance series
    smooth_points: int = 3

    # detection thresholds (in "sigma" units of robust SNR)
    detect_thr: float = 3.0          # must exceed somewhere to declare event
    window_thr: float = 1.0          # expand window while above this (after smoothing)

    # require at least this many points above detect_thr in chosen band
    min_points_above: int = 2

    # expansion in time around detected contiguous region
    pad_days: float = 0.0

    # fallback behavior
    fallback_to_full: bool = True


def _robust_snr(flux: np.ndarray, err: np.ndarray) -> np.ndarray:
    """SNR with NaNs where err is non-finite or <= 0."""
    snr = np.full_like(flux, np.nan, dtype=float)
    m = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    snr[m] = flux[m] / err[m]
    snr[~np.isfinite(snr)] = np.nan
    return snr


def _rolling_mean_nan(x: np.ndarray, k: int) -> np.ndarray:
    """Simple centered rolling mean that ignores NaNs; k>=1."""
    if k <= 1:
        return x.astype(float, copy=True)
    x = x.astype(float, copy=False)
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    half = k // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = x[lo:hi]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = w.mean()
    return out


def _robust_baseline_scale(flux: np.ndarray) -> Tuple[float, float]:
    """Median + MAD (scaled) baseline/scale. Returns (median, sigma_robust)."""
    finite = flux[np.isfinite(flux)]
    if finite.size == 0:
        return (np.nan, np.nan)
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    sigma = 1.4826 * mad  # normal-equivalent
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nan
    return (float(med), float(sigma))


def _choose_best_band(g: pd.DataFrame, params: EventWindowParams) -> str:
    """
    Choose the band with the strongest positive transient-like excursion:
    max of smoothed robust significance (positive side).
    """
    best_band = None
    best_score = -np.inf

    for flt in params.filters:
        gf = g[g["Filter"] == flt]
        if gf.empty:
            continue

        f = gf["Flux"].to_numpy(dtype=float)
        e = gf["Flux_err"].to_numpy(dtype=float)

        med, sigma = _robust_baseline_scale(f)
        if not np.isfinite(med):
            continue

        # prefer using Flux_err when sane; otherwise fall back to robust sigma
        snr_err = _robust_snr(f - med, e)
        if np.isfinite(snr_err).any():
            s = snr_err
        elif np.isfinite(sigma) and sigma > 0:
            s = (f - med) / sigma
            s[~np.isfinite(s)] = np.nan
        else:
            continue

        s_smooth = _rolling_mean_nan(s, params.smooth_points)
        score = np.nanmax(s_smooth) if np.isfinite(s_smooth).any() else -np.inf
        if score > best_score:
            best_score = score
            best_band = flt

    return best_band if best_band is not None else str(params.filters[0])


def detect_event_window_for_object(
    g: pd.DataFrame,
    params: EventWindowParams = EventWindowParams(),
) -> Dict[str, object]:
    """
    Detect an event window for a single object's light curve.

    Returns dict with:
      - 'chosen_filter'
      - 't_start', 't_end' (float or NaN)
      - 'has_event' (bool)
      - 'reason' (str)
      - 'mask' (np.ndarray[bool] same length as g, aligned to g.index order)
      - 'peak_time' (float or NaN) in chosen band (time of max smoothed significance)
    """
    if g.empty:
        return {
            "chosen_filter": None,
            "t_start": np.nan,
            "t_end": np.nan,
            "has_event": False,
            "reason": "empty",
            "mask": np.zeros(0, dtype=bool),
            "peak_time": np.nan,
        }

    # Ensure sorted by time but keep original row order for mask alignment
    g_sorted = g.sort_values("Time (MJD)")
    idx_sorted = g_sorted.index.to_numpy()

    chosen = _choose_best_band(g_sorted, params)
    gf = g_sorted[g_sorted["Filter"] == chosen].copy()

    t = gf["Time (MJD)"].to_numpy(dtype=float)
    f = gf["Flux"].to_numpy(dtype=float)
    e = gf["Flux_err"].to_numpy(dtype=float)

    med, sigma = _robust_baseline_scale(f)
    snr_err = _robust_snr(f - med, e) if np.isfinite(med) else np.full_like(f, np.nan, dtype=float)

    if np.isfinite(snr_err).any():
        s = snr_err
        s_kind = "err"
    elif np.isfinite(sigma) and sigma > 0:
        s = (f - med) / sigma
        s[~np.isfinite(s)] = np.nan
        s_kind = "mad"
    else:
        if params.fallback_to_full:
            t_all = g_sorted["Time (MJD)"].to_numpy(dtype=float)
            t0 = float(np.nanmin(t_all)) if np.isfinite(t_all).any() else np.nan
            t1 = float(np.nanmax(t_all)) if np.isfinite(t_all).any() else np.nan
            mask_full = np.ones(len(g_sorted), dtype=bool)
            mask = pd.Series(mask_full, index=idx_sorted).reindex(g.index, fill_value=False).to_numpy(bool)
            return {
                "chosen_filter": chosen,
                "t_start": t0,
                "t_end": t1,
                "has_event": False,
                "reason": "no_valid_significance_fallback_full",
                "mask": mask,
                "peak_time": np.nan,
            }
        return {
            "chosen_filter": chosen,
            "t_start": np.nan,
            "t_end": np.nan,
            "has_event": False,
            "reason": "no_valid_significance",
            "mask": np.zeros(len(g), dtype=bool),
            "peak_time": np.nan,
        }

    s_smooth = _rolling_mean_nan(s, params.smooth_points)

    # declare detection if enough points above detect_thr (raw or smooth)
    above = np.isfinite(s_smooth) & (s_smooth >= params.detect_thr)
    if above.sum() < params.min_points_above:
        if params.fallback_to_full:
            t_all = g_sorted["Time (MJD)"].to_numpy(dtype=float)
            t0 = float(np.nanmin(t_all)) if np.isfinite(t_all).any() else np.nan
            t1 = float(np.nanmax(t_all)) if np.isfinite(t_all).any() else np.nan
            mask_full = np.ones(len(g_sorted), dtype=bool)
            mask = pd.Series(mask_full, index=idx_sorted).reindex(g.index, fill_value=False).to_numpy(bool)
            return {
                "chosen_filter": chosen,
                "t_start": t0,
                "t_end": t1,
                "has_event": False,
                "reason": f"no_detection_{s_kind}_fallback_full",
                "mask": mask,
                "peak_time": np.nan,
            }
        return {
            "chosen_filter": chosen,
            "t_start": np.nan,
            "t_end": np.nan,
            "has_event": False,
            "reason": f"no_detection_{s_kind}",
            "mask": np.zeros(len(g), dtype=bool),
            "peak_time": np.nan,
        }

    # peak index in chosen band (smoothed)
    peak_i = int(np.nanargmax(s_smooth))
    peak_t = float(t[peak_i]) if np.isfinite(t[peak_i]) else np.nan

    # expand window around peak while above window_thr (smoothed)
    inwin = np.isfinite(s_smooth) & (s_smooth >= params.window_thr)
    # find contiguous segment containing peak_i
    left = peak_i
    right = peak_i
    while left - 1 >= 0 and inwin[left - 1]:
        left -= 1
    while right + 1 < len(inwin) and inwin[right + 1]:
        right += 1

    t0 = float(t[left]) if np.isfinite(t[left]) else np.nan
    t1 = float(t[right]) if np.isfinite(t[right]) else np.nan

    # optional padding in time
    if np.isfinite(t0) and np.isfinite(t1) and params.pad_days > 0:
        t0 -= params.pad_days
        t1 += params.pad_days

    # Build mask on the full (sorted) object table: include all bands within [t0, t1]
    t_all = g_sorted["Time (MJD)"].to_numpy(dtype=float)
    mask_sorted = np.isfinite(t_all) & (t_all >= t0) & (t_all <= t1)

    # map back to original g order
    mask = pd.Series(mask_sorted, index=idx_sorted).reindex(g.index, fill_value=False).to_numpy(bool)

    return {
        "chosen_filter": chosen,
        "t_start": t0,
        "t_end": t1,
        "has_event": True,
        "reason": f"detected_{s_kind}",
        "mask": mask,
        "peak_time": peak_t,
    }


def detect_event_windows(
    lcs: pd.DataFrame,
    object_ids: Optional[Iterable[str]] = None,
    params: EventWindowParams = EventWindowParams(),
) -> pd.DataFrame:
    """
    Batch detect event windows. Returns one row per object_id:
      object_id, chosen_filter, t_start, t_end, has_event, reason, peak_time
    """
    if object_ids is not None:
        obj_set = set(object_ids)
        df = lcs[lcs["object_id"].isin(obj_set)].copy()
    else:
        df = lcs.copy()

    rows = []
    for oid, g in df.groupby("object_id", sort=False):
        res = detect_event_window_for_object(g, params=params)
        rows.append({
            "object_id": oid,
            "chosen_filter": res["chosen_filter"],
            "t_start": res["t_start"],
            "t_end": res["t_end"],
            "peak_time": res["peak_time"],
            "has_event": res["has_event"],
            "reason": res["reason"],
        })
    return pd.DataFrame(rows)
