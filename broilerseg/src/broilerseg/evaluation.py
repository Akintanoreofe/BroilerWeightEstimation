"""Weight-error metrics, event-level aggregation and density x slice-count sweeps.

Event-level numbers follow the repository's ``tune_validation.py``: for each
weighing event, compare the median estimate with the median scale reading.
Fit density on *calibration* events and report error on *held-out* events,
otherwise the error is optimistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from broilerseg.volume import mass_from_volume


def regression_metrics(actual, predicted) -> dict:
    """MAE, RMSE, bias (pred - actual), MAPE (%), R^2 and n for paired arrays."""
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if len(a) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "mape": np.nan,
                "r2": np.nan}
    e = p - a
    nz = a != 0
    ss_tot = np.sum((a - a.mean()) ** 2)
    return {"n": int(len(a)), "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)),
            "mape": float(np.mean(np.abs(e[nz] / a[nz])) * 100) if nz.any() else np.nan,
            "r2": float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else np.nan}


def event_table(records: pd.DataFrame, pred_col: str = "pred_kg",
                actual_col: str = "actual_kg") -> pd.DataFrame:
    """Median prediction vs median scale reading per event (valid frames only)."""
    ok = records[np.isfinite(records[pred_col]) & np.isfinite(records[actual_col])]
    t = (ok.groupby("event_id")
         .agg(video_name=("video_name", "first"), actual_kg=(actual_col, "median"),
              pred_kg=(pred_col, "median"), pred_std_kg=(pred_col, "std"),
              n_frames=(pred_col, "size"))
         .reset_index())
    t["error_kg"] = t.pred_kg - t.actual_kg
    t["pct_error"] = 100 * t.error_kg / t.actual_kg
    return t


def split_events(event_ids, calibration_fraction: float = 0.5, seed: int = 0
                 ) -> tuple[list[str], list[str]]:
    """Randomly split event ids into disjoint calibration and held-out lists."""
    ids = sorted({str(e) for e in event_ids})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_cal = int(round(len(ids) * calibration_fraction))
    if len(ids) >= 2:
        n_cal = min(max(n_cal, 1), len(ids) - 1)
    return sorted(ids[:n_cal]), sorted(ids[n_cal:])


def _event_errors(df: pd.DataFrame, density: float, ids) -> np.ndarray:
    sub = df[df.event_id.isin(ids)]
    if sub.empty:
        return np.array([])
    g = sub.groupby("event_id").agg(v=("volume_cm3", "median"), a=("actual_kg", "median"))
    return mass_from_volume(g.v.to_numpy(), density) - g.a.to_numpy()


def sweep_grid(volume_table: pd.DataFrame, densities, calibration_events,
               heldout_events) -> pd.DataFrame:
    """Event-level error for every (slice count, density) pair.

    Parameters
    ----------
    volume_table : pandas.DataFrame
        Columns ``event_id``, ``slice_count``, ``volume_cm3``, ``actual_kg``
        (one row per frame and slice count), e.g. from
        :meth:`broilerseg.experiments.WeightExperiment.volume_table`.
    densities : iterable of float
        Densities in kg / m^3.
    calibration_events, heldout_events : list of str
        Disjoint event ids.
    """
    cal, held = set(map(str, calibration_events)), set(map(str, heldout_events))
    if cal & held:
        raise ValueError("Calibration and held-out events must be disjoint")
    rows = []
    vt = volume_table.assign(event_id=volume_table.event_id.astype(str))
    for sc, df in vt.groupby("slice_count", sort=False):
        for rho in densities:
            row = {"slice_count": sc, "density_kg_m3": float(rho)}
            for tag, ids in (("cal", cal), ("held", held)):
                e = _event_errors(df, rho, ids)
                row[f"{tag}_n_events"] = len(e)
                row[f"{tag}_rmse_kg"] = float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan
                row[f"{tag}_mae_kg"] = float(np.mean(np.abs(e))) if len(e) else np.nan
                row[f"{tag}_bias_kg"] = float(np.mean(e)) if len(e) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def best_density(volume_table: pd.DataFrame, calibration_events, heldout_events=()
                 ) -> pd.DataFrame:
    """Least-squares density per slice count, fitted on calibration events.

    Because ``median(V * rho) = rho * median(V)``, the density that minimises
    event-level squared error has the closed form
    ``rho = sum(V_m3 * actual) / sum(V_m3 ** 2)``.
    """
    cal = set(map(str, calibration_events))
    rows = []
    vt = volume_table.assign(event_id=volume_table.event_id.astype(str))
    for sc, df in vt.groupby("slice_count", sort=False):
        g = (df[df.event_id.isin(cal)].groupby("event_id")
             .agg(v=("volume_cm3", "median"), a=("actual_kg", "median")))
        v = g.v.to_numpy() / 1e6
        rho = float(np.sum(v * g.a.to_numpy()) / np.sum(v ** 2)) if len(v) else np.nan
        cal_e = _event_errors(df, rho, cal) if np.isfinite(rho) else np.array([])
        held_e = _event_errors(df, rho, heldout_events) if np.isfinite(rho) else np.array([])
        rows.append({"slice_count": sc, "best_density_kg_m3": rho,
                     "cal_rmse_kg": float(np.sqrt(np.mean(cal_e ** 2))) if len(cal_e) else np.nan,
                     "held_rmse_kg": float(np.sqrt(np.mean(held_e ** 2))) if len(held_e)
                     else np.nan,
                     "held_mae_kg": float(np.mean(np.abs(held_e))) if len(held_e) else np.nan})
    return pd.DataFrame(rows)
