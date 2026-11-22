r"""One mass equation with every parameter fitted to the scale readings.

The slice (stacked-cylinder) model of the repository is extended so that each
known shortcoming has a parameter that the data can correct:

.. math::

    \hat m \;=\; a\,\Big[\tfrac{\pi}{4}\sum_{i\in S_\tau} \big(p_c\,d_i^{(\delta)}\big)^2\,
    p_c\,w_i\Big]^{\,b} \;=\; a\,V^{b}

========================  =====================================================
symbol                    shortcoming it corrects
========================  =====================================================
:math:`\delta` (px)       mask boundary too large / small (feathers, soft edges);
                          the mask is dilated (+) or eroded (-) before slicing
:math:`\tau`              head, neck and tail counted as solid body: a fraction
                          of the length is trimmed at each end
:math:`n`                 number of slices (``None`` = one per pixel column)
:math:`c`                 calibration method + scope used for :math:`p_c`
                          (cm / px): frame, video or camera median
:math:`b`                 weight is not exactly proportional to volume
                          (size-dependent bias; allometric exponent)
:math:`a`                 effective density (for :math:`b = 1`,
                          :math:`\rho = a \cdot 10^6` kg / m^3)
========================  =====================================================

Fitting
-------
* :math:`a, b` have a closed form: weighted least squares of
  :math:`\log m = \log a + b \log V`, with a ridge prior pulling :math:`b`
  towards 1. Each event gets equal weight. When the scale readings barely
  vary (coefficient of variation below ``min_weight_cv``), :math:`b` is not
  identifiable and is fixed at 1.
* :math:`\delta, \tau, n, c` are searched on a grid.
* Every configuration is scored by **leave-one-event-out** cross-validation
  on the event-level error (median estimate vs median scale reading), so the
  reported error is never measured on the data used to fit it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from broilerseg.calibration import METHODS
from broilerseg.config import FEEDER_DIAMETER_CM
from broilerseg.volume import VolumeEstimate, align_mask, estimate_volume, mask_slice_profile

SCOPES = ("frame", "video", "camera")


# ----------------------------------------------------------------------- geometry
def adjust_mask(mask: np.ndarray, offset_px: int) -> np.ndarray:
    """Dilate (``offset_px > 0``) or erode (``< 0``) a mask by a disc of that radius."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if offset_px == 0:
        return m.astype(bool)
    r = abs(int(offset_px))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = cv2.dilate(m, k) if offset_px > 0 else cv2.erode(m, k)
    return out.astype(bool)


def trim_ends(aligned: np.ndarray, trim: float) -> np.ndarray:
    """Remove a fraction ``trim`` of the occupied length at each end of an aligned mask."""
    a = np.asarray(aligned) > 0
    if trim <= 0:
        return a
    cols = np.flatnonzero(a.any(axis=0))
    k = int(round(trim * len(cols)))
    if k == 0 or 2 * k >= len(cols):
        return a
    out = a.copy()
    out[:, :cols[k]] = False
    out[:, cols[-1 - k] + 1:] = False
    return out


def _volume_px_from_aligned(aligned: np.ndarray, trim: float, slice_count) -> float:
    prof = mask_slice_profile(trim_ends(aligned, trim), slice_count)
    if not len(prof["thickness_px"]):
        return 0.0
    d = (prof["y_max"] - prof["y_min"] + 1).astype(float)
    return float(np.pi / 4 * np.sum(d ** 2 * prof["thickness_px"]))


def shape_volume_px(mask: np.ndarray, offset_px: int = 0, trim: float = 0.0,
                    slice_count: int | None = None) -> float:
    """Slice volume in **pixel** units (px^3); multiply by ``pixel_to_cm**3`` for cm^3."""
    aligned, _ = align_mask(adjust_mask(mask, offset_px))
    return _volume_px_from_aligned(aligned, trim, slice_count)


def _slice_label(k) -> str | int:
    return "per-column" if k is None else int(k)


def _slice_value(label):
    return None if label == "per-column" else int(label)


# ----------------------------------------------------------------------- the model
@dataclass
class MassModel:
    r"""The fitted equation :math:`\hat m = a V^b` and its shape / calibration settings.

    Use :meth:`predict` on a bird mask with a pixel scale measured by the same
    calibration method (``calib_method``).
    """

    a: float
    b: float
    offset_px: int = 0
    trim: float = 0.0
    slice_count: int | None = None
    calib_method: str = "enclosing_circle"
    calib_scope: str = "video"

    def volume_cm3(self, mask: np.ndarray, pixel_to_cm: float) -> float:
        """Return the adjusted slice volume of one mask in cm^3."""
        return shape_volume_px(mask, self.offset_px, self.trim, self.slice_count) \
            * pixel_to_cm ** 3

    def predict(self, mask: np.ndarray, pixel_to_cm: float) -> float:
        """Estimated mass in kg."""
        v = self.volume_cm3(mask, pixel_to_cm)
        return float(self.a * v ** self.b) if v > 0 else float("nan")

    def estimate(self, mask: np.ndarray, pixel_to_cm: float) -> VolumeEstimate:
        """Full :class:`VolumeEstimate` (for plots) with the fitted settings and mass.

        ``density_kg_m3`` of the result is the density the fitted equation
        implies for this bird.
        """
        aligned, angle = align_mask(adjust_mask(mask, self.offset_px))
        trimmed = trim_ends(aligned, self.trim).astype(np.uint8) * 255
        est = estimate_volume(trimmed, pixel_to_cm, 1000.0, self.slice_count, align=False)
        est.angle_deg = angle
        est.mass_kg = float(self.predict_from_volume(est.volume_cm3))
        est.density_kg_m3 = est.mass_kg / (est.volume_cm3 / 1e6) if est.volume_cm3 > 0 \
            else float("nan")
        return est

    def predict_from_volume(self, volume_cm3):
        """Mass (kg) from adjusted volume(s) in cm^3."""
        return self.a * np.asarray(volume_cm3, float) ** self.b

    def effective_density(self, volume_cm3: float | None = None) -> float:
        """Density (kg/m^3) implied by the model; exact for ``b = 1``.

        For ``b != 1`` the implied density depends on size and is reported at
        ``volume_cm3``.
        """
        if abs(self.b - 1) < 1e-9 or volume_cm3 is None:
            return self.a * 1e6 if abs(self.b - 1) < 1e-9 else float("nan")
        return self.a * volume_cm3 ** (self.b - 1) * 1e6

    def equation(self, unit: str = "kg") -> str:
        """Human-readable equation with the fitted numbers."""
        f = 1000 if unit == "g" else 1
        sc = "one per pixel column" if self.slice_count is None else f"{self.slice_count}"
        return (f"m [{unit}] = {self.a * f:.5g} * V^{self.b:.3f}   with V = pi/4 * sum (p*d_i)^2 "
                f"* p*w_i  [cm^3];  boundary offset {self.offset_px:+d} px, end trim "
                f"{100 * self.trim:.0f}% each end, slices: {sc}, p from {self.calib_method} "
                f"({self.calib_scope} median)")

    def as_dict(self) -> dict:
        """Parameters as a flat dictionary."""
        return {"a": self.a, "b": self.b, "offset_px": self.offset_px, "trim": self.trim,
                "slice_count": _slice_label(self.slice_count),
                "calib_method": self.calib_method, "calib_scope": self.calib_scope,
                "effective_density_kg_m3": float(self.effective_density())}


# ----------------------------------------------------------------------- features
def build_feature_table(experiment, offsets=(-4, -2, 0, 2, 4), trims=(0.0, 0.05, 0.10, 0.15),
                        slice_counts=(None, 5, 10, 15, 20, 30), progress: bool = True
                        ) -> pd.DataFrame:
    """Slice volume (px^3) of every bird for every (offset, trim, slice count).

    ``experiment`` is a :class:`broilerseg.experiments.WeightExperiment` that
    has been run (its cleaned bird masks are reused; nothing is re-segmented).
    """
    rec = experiment.records
    ids = [int(s) for s in rec.sample_id if int(s) in experiment.birds]
    meta = rec.set_index("sample_id")
    rows = []
    it = ids
    if progress:
        try:
            from tqdm.auto import tqdm

            it = tqdm(ids, desc="Shape features")
        except ImportError:  # pragma: no cover
            pass
    for sid in it:
        mask = experiment.birds[sid].mask
        m = meta.loc[sid]
        for off in offsets:
            aligned, _ = align_mask(adjust_mask(mask, off))
            for tr in trims:
                for k in slice_counts:
                    rows.append({"sample_id": sid, "event_id": str(m.event_id),
                                 "video_name": m.video_name, "actual_kg": float(m.actual_kg),
                                 "offset_px": int(off), "trim": float(tr),
                                 "slice_count": _slice_label(k),
                                 "v_px": _volume_px_from_aligned(aligned, tr, k)})
    return pd.DataFrame(rows)


def build_scale_table(experiment, methods=METHODS, scopes=SCOPES, camera_col: str | None = None,
                      diameter_cm: float = FEEDER_DIAMETER_CM) -> pd.DataFrame:
    """Pixel scale of every sample for every calibration method and scope.

    ``frame`` uses the frame's own feeder, ``video`` the median over the
    video, ``camera`` the median over all frames of the camera (``camera_col``
    in the records, or the whole dataset when it is ``None``: right for one
    fixed camera, as in a FlockLabel export with one crop).
    """
    rec = experiment.records[["sample_id", "video_name"]].copy()
    rec["camera"] = experiment.records[camera_col] if camera_col else "all"
    rows = []
    for method in methods:
        px = []
        for sid in rec.sample_id:
            cal = experiment.calibrations.get(int(sid))
            d = cal.diameters_px.get(method, np.nan) if cal is not None else np.nan
            px.append(diameter_cm / d if np.isfinite(d) and d > 10 else np.nan)
        r = rec.assign(frame=px)
        r["video"] = r.groupby("video_name").frame.transform("median")
        r["camera"] = r.groupby("camera").frame.transform("median")
        for scope in scopes:
            rows.append(pd.DataFrame({"sample_id": r.sample_id.astype(int), "calib_method": method,
                                      "calib_scope": scope, "pixel_to_cm": r[scope]}))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------- fitting
def fit_ab(volume_cm3, mass_kg, weights=None, fix_b: bool = False, ridge: float = 0.1
           ) -> tuple[float, float]:
    """Closed-form weighted fit of ``log m = log a + b log V`` with a ridge prior b -> 1.

    Minimises ``sum w (log m - log a - b log V)^2 + ridge * sum(w) * (b - 1)^2``.
    """
    v = np.asarray(volume_cm3, float)
    m = np.asarray(mass_kg, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(m) & (v > 0) & (m > 0) & (w > 0)
    x, y, w = np.log(v[ok]), np.log(m[ok]), w[ok]
    if len(x) == 0:
        return float("nan"), float("nan")
    sw = w.sum()
    if fix_b:
        return float(np.exp(np.sum(w * (y - x)) / sw)), 1.0
    lam = ridge * sw
    A = np.array([[sw, np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x) + lam]])
    rhs = np.array([np.sum(w * y), np.sum(w * x * y) + lam])
    alpha, b = np.linalg.solve(A, rhs)
    return float(np.exp(alpha)), float(b)


def _event_errors(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    t = df.assign(pred_kg=pred).groupby("event_id").agg(
        actual_kg=("actual_kg", "median"), pred_kg=("pred_kg", "median"),
        n_frames=("pred_kg", "size"))
    t["error_kg"] = t.pred_kg - t.actual_kg
    return t


def _metrics(err: np.ndarray, actual: np.ndarray) -> dict:
    err, actual = np.asarray(err, float), np.asarray(actual, float)
    ok = np.isfinite(err)
    if not ok.any():
        return {"rmse_kg": np.nan, "mae_kg": np.nan, "bias_kg": np.nan, "mape": np.nan}
    e, a = err[ok], actual[ok]
    return {"rmse_kg": float(np.sqrt(np.mean(e ** 2))), "mae_kg": float(np.mean(np.abs(e))),
            "bias_kg": float(np.mean(e)), "mape": float(np.mean(np.abs(e / a)) * 100)}


@dataclass
class FitReport:
    """Everything produced by :func:`fit_mass_model` (all errors are out-of-fold)."""

    model: MassModel
    cv_table: pd.DataFrame = field(repr=False)
    oof_events: pd.DataFrame = field(repr=False)
    oof_frames: pd.DataFrame = field(repr=False)
    folds: pd.DataFrame = field(repr=False)
    comparison: pd.DataFrame = field(repr=False)
    fix_b: bool = False
    weight_cv: float = float("nan")
    n_events: int = 0
    b_searched: bool = True
    nested_folds: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    data: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def b_status(self) -> str:
        """Why the exponent has its value."""
        if not self.b_searched:
            return "fixed at 1: event weights too similar to learn a size exponent"
        if self.fix_b:
            return "1 chosen by cross-validation (fitting b did not reduce the error)"
        return "fitted"

    def summary(self) -> pd.Series:
        """Best configuration and its cross-validated error."""
        best = self.cv_table.iloc[0]
        med_v = float(self.data.volume_cm3.median()) if len(self.data) else None
        return pd.Series({**self.model.as_dict(),
                          "density_at_median_volume_kg_m3": self.model.effective_density(med_v),
                          "cv_rmse_kg": best.cv_rmse_kg,
                          "cv_mae_kg": best.cv_mae_kg, "cv_mape": best.cv_mape,
                          "b_status": self.b_status(), "events": self.n_events,
                          "weight_cv_percent": 100 * self.weight_cv})


class _Arrays:
    """Features and scales as aligned numpy arrays, for a fast grid search."""

    def __init__(self, features: pd.DataFrame, scales: pd.DataFrame):
        ids = np.array(sorted(features.sample_id.unique()))
        self.ids = ids
        meta = features.drop_duplicates("sample_id").set_index("sample_id").loc[ids]
        self.event = meta.event_id.astype(str).to_numpy()
        self.actual = meta.actual_kg.to_numpy(float)
        self.video = meta.video_name.to_numpy()
        key = features.set_index(["offset_px", "trim", "slice_count", "sample_id"]).v_px
        self.v_px = {k: g.droplevel([0, 1, 2]).reindex(ids).to_numpy(float)
                     for k, g in key.groupby(level=[0, 1, 2], sort=False)}
        sk = scales.set_index(["calib_method", "calib_scope", "sample_id"]).pixel_to_cm
        self.px = {k: g.droplevel([0, 1]).reindex(ids).to_numpy(float)
                   for k, g in sk.groupby(level=[0, 1], sort=False)}
        self.events = np.array(sorted(set(self.event)))

    def volume(self, shape_key, scale_key) -> np.ndarray:
        return self.v_px[shape_key] * self.px[scale_key] ** 3

    def frame(self, shape_key, scale_key) -> pd.DataFrame:
        v = self.volume(shape_key, scale_key)
        ok = np.isfinite(v) & (v > 0) & np.isfinite(self.actual)
        d = pd.DataFrame({"sample_id": self.ids[ok], "event_id": self.event[ok],
                          "video_name": self.video[ok], "actual_kg": self.actual[ok],
                          "pixel_to_cm": self.px[scale_key][ok], "volume_cm3": v[ok]})
        d["w"] = 1.0 / d.groupby("event_id").sample_id.transform("size")
        return d


def _cv_arrays(v, m, ev, events, fix_b, ridge, a_fixed=None):
    """Leave-one-event-out predictions (numpy) and per-fold parameters."""
    n_per = {e: np.sum(ev == e) for e in events}
    w = np.array([1.0 / n_per[e] for e in ev])
    pred = np.full(len(v), np.nan)
    folds = []
    for e in events:
        test = ev == e
        if a_fixed is None:
            a, b = fit_ab(v[~test], m[~test], w[~test], fix_b, ridge)
        else:
            a, b = a_fixed, 1.0
        pred[test] = a * v[test] ** b
        folds.append({"held_out_event": e, "a": a, "b": b,
                      "density_kg_m3": a * 1e6 if abs(b - 1) < 1e-9 else np.nan})
    return pred, folds


def _event_metrics(pred, m, ev, events) -> dict:
    err, act = [], []
    for e in events:
        k = ev == e
        err.append(np.median(pred[k]) - np.median(m[k]))
        act.append(np.median(m[k]))
    return _metrics(np.array(err), np.array(act))


def _evaluate(arr: _Arrays, shape_key, scale_key, fix_b, ridge, a_fixed=None):
    v = arr.volume(shape_key, scale_key)
    ok = np.isfinite(v) & (v > 0) & np.isfinite(arr.actual)
    v, m, ev = v[ok], arr.actual[ok], arr.event[ok]
    events = np.array(sorted(set(ev)))
    pred, folds = _cv_arrays(v, m, ev, events, fix_b, ridge, a_fixed)
    w = np.array([1.0 / np.sum(ev == e) for e in ev])
    a, b = fit_ab(v, m, w, fix_b, ridge) if a_fixed is None else (a_fixed, 1.0)
    return _event_metrics(pred, m, ev, events), pred, folds, (a, b), len(events)


def _grid_search(arr: _Arrays, b_modes, ridge: float, progress: bool = False,
                 desc: str = "") -> pd.DataFrame:
    """Score every configuration by leave-one-event-out CV; best first."""
    n_events = len(arr.events)
    grid = list(itertools.product(arr.v_px.keys(), arr.px.keys(), b_modes))
    it = grid
    if progress:
        try:
            from tqdm.auto import tqdm

            it = tqdm(grid, desc=desc)
        except ImportError:  # pragma: no cover
            pass
    rows = []
    for shape_key, scale_key, fb in it:
        met, _, _, (a, b), n_ev = _evaluate(arr, shape_key, scale_key, fb, ridge)
        if n_ev < n_events:
            continue  # a calibration failed for some event: not comparable
        off, tr, sc = shape_key
        rows.append({"offset_px": off, "trim": tr, "slice_count": sc,
                     "calib_method": scale_key[0], "calib_scope": scale_key[1],
                     "b_mode": "b = 1" if fb else "b fitted", "a": a, "b": b,
                     **{f"cv_{k}": v for k, v in met.items()}})
    if not rows:
        raise ValueError("No configuration covered every event (check calibrations)")
    cv = pd.DataFrame(rows)
    # among (numerically) equal errors prefer the simplest settings
    cv["_simple"] = (cv.offset_px.abs() + 10 * cv.trim + (cv.slice_count != "per-column") * 0.1
                     + (cv.b_mode == "b fitted") * 0.5)
    cv["_rmse"] = cv.cv_rmse_kg.round(6)
    cv = cv.sort_values(["_rmse", "_simple"]).drop(columns=["_simple", "_rmse"])
    return cv.reset_index(drop=True)


def _nested_cv(features, scales, b_modes, ridge, progress) -> tuple[dict, pd.DataFrame]:
    """Re-run the whole search without each event, then predict that event.

    This measures the error of the complete procedure (search + fit), so it
    is not flattered by having picked the best of many configurations.
    """
    full = _Arrays(features, scales)
    events = list(full.events)
    it = events
    if progress:
        try:
            from tqdm.auto import tqdm

            it = tqdm(events, desc="Nested cross-validation")
        except ImportError:  # pragma: no cover
            pass
    rows, err, act = [], [], []
    for e in it:
        inner = _Arrays(features[features.event_id.astype(str) != e], scales)
        if len(inner.events) < 2:
            continue
        best = _grid_search(inner, b_modes, ridge).iloc[0]
        shape_key = (best.offset_px, best.trim, best.slice_count)
        scale_key = (best.calib_method, best.calib_scope)
        v = full.volume(shape_key, scale_key)
        test = (full.event == e) & np.isfinite(v) & (v > 0)
        pred = best.a * v[test] ** best.b
        e_err = float(np.median(pred) - np.median(full.actual[test])) if test.any() else np.nan
        err.append(e_err)
        act.append(float(np.median(full.actual[full.event == e])))
        rows.append({"held_out_event": e, "offset_px": best.offset_px, "trim": best.trim,
                     "slice_count": best.slice_count, "calib_method": best.calib_method,
                     "calib_scope": best.calib_scope, "b_mode": best.b_mode, "a": best.a,
                     "b": best.b, "error_kg": e_err, "actual_kg": act[-1]})
    return _metrics(np.array(err), np.array(act)), pd.DataFrame(rows)


def fit_mass_model(features: pd.DataFrame, scales: pd.DataFrame, fix_b: bool | None = None,
                   ridge: float = 0.1, min_weight_cv: float = 0.05,
                   baseline_density: float = 900.0, progress: bool = True,
                   nested: bool = True) -> FitReport:
    """Search every configuration and fit ``a, b`` with leave-one-event-out CV.

    Parameters
    ----------
    features, scales : pandas.DataFrame
        From :func:`build_feature_table` and :func:`build_scale_table`.
    fix_b : bool, optional
        Force ``b = 1``. ``None`` decides from the data: ``b`` is fixed when the
        event weights vary by less than ``min_weight_cv`` (coefficient of
        variation), because a size exponent cannot be learned from birds of one
        weight. Otherwise both ``b = 1`` and a free ``b`` are tried for every
        configuration and the better (or, on a tie, the simpler) is kept.
    ridge : float
        Strength of the prior pulling ``b`` towards 1 (0 = plain least squares).
    min_weight_cv : float
        Weight variation below which ``b`` is fixed at 1 (when ``fix_b`` is None).
    baseline_density : float
        Density of the un-fitted repository baseline shown in the comparison.
    progress : bool
        Show progress bars.
    nested : bool
        Also run nested cross-validation (the whole search is repeated without
        each event). Gives an unbiased error for the complete procedure and
        shows whether the chosen settings are stable. Costs one extra search
        per event.
    """
    arr = _Arrays(features, scales)
    n_events = len(arr.events)
    if n_events < 2:
        raise ValueError("Need at least 2 weighing events for cross-validation")
    ev_w = pd.Series(arr.actual).groupby(arr.event).median()
    weight_cv = float(ev_w.std(ddof=0) / ev_w.mean()) if ev_w.mean() > 0 else 0.0
    if fix_b is None:
        fix_b = weight_cv < min_weight_cv
    b_modes = (True,) if fix_b else (True, False)
    cv = _grid_search(arr, b_modes, ridge, progress, "Fitting configurations")
    best = cv.iloc[0]
    best_fix = best.b_mode == "b = 1"
    model = MassModel(float(best.a), float(best.b), int(best.offset_px), float(best.trim),
                      _slice_value(best.slice_count), str(best.calib_method),
                      str(best.calib_scope))
    shape_key = (best.offset_px, best.trim, best.slice_count)
    scale_key = (best.calib_method, best.calib_scope)
    d = arr.frame(shape_key, scale_key)
    pred, folds = _cv_arrays(d.volume_cm3.to_numpy(), d.actual_kg.to_numpy(),
                             d.event_id.to_numpy(), np.array(sorted(d.event_id.unique())),
                             best_fix, ridge)
    oof_frames = d.assign(pred_kg=pred, error_kg=pred - d.actual_kg)
    oof_events = _event_errors(d, pred).reset_index()
    comparison = _comparison(arr, shape_key, scale_key, fix_b, ridge, baseline_density)
    nested_folds = pd.DataFrame()
    if nested and n_events >= 3:
        met, nested_folds = _nested_cv(features, scales, b_modes, ridge, progress)
        comparison = pd.concat([comparison, pd.DataFrame(
            [{"model": "full search, nested CV (unbiased)", **met}])], ignore_index=True)
    return FitReport(model, cv, oof_events, oof_frames, pd.DataFrame(folds), comparison,
                     bool(best_fix), weight_cv, n_events, data=d, b_searched=not fix_b,
                     nested_folds=nested_folds)


def _comparison(arr: _Arrays, shape_key, scale_key, fix_b, ridge, baseline_density):
    """Repository baseline vs density-only fit vs the full model (all out-of-fold)."""
    out = []
    base_shape, base_scale = (0, 0.0, "per-column"), ("enclosing_circle", "video")
    if base_shape in arr.v_px and base_scale in arr.px:
        met = _evaluate(arr, base_shape, base_scale, True, ridge, baseline_density / 1e6)[0]
        out.append({"model": f"repository (density {baseline_density:.0f}, not fitted)", **met})
        met = _evaluate(arr, base_shape, base_scale, True, ridge)[0]
        out.append({"model": "density fitted only", **met})
    met = _evaluate(arr, shape_key, scale_key, True, ridge)[0]
    out.append({"model": "density + shape + calibration (b = 1)", **met})
    if not fix_b:
        met = _evaluate(arr, shape_key, scale_key, False, ridge)[0]
        out.append({"model": "full equation (b fitted)", **met})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------- density
def estimate_density(report: FitReport, n_boot: int = 2000, seed: int = 0) -> dict:
    """Most accurate single density (``b = 1``) for the selected configuration.

    Returns the density fitted on all events, the leave-one-event-out values
    (how much it moves when one event is left out), and a bootstrap 95%
    interval obtained by resampling whole events.
    """
    d = report.data
    a, _ = fit_ab(d.volume_cm3, d.actual_kg, d.w, fix_b=True)
    fold = []
    for e in sorted(d.event_id.unique()):
        tr = d[d.event_id != e]
        fold.append(fit_ab(tr.volume_cm3, tr.actual_kg, tr.w, fix_b=True)[0] * 1e6)
    rng = np.random.default_rng(seed)
    events = d.event_id.unique()
    groups = {e: d[d.event_id == e] for e in events}
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(events, size=len(events), replace=True)
        bd = pd.concat([groups[e] for e in pick], ignore_index=True)
        boot.append(fit_ab(bd.volume_cm3, bd.actual_kg, bd.w, fix_b=True)[0] * 1e6)
    boot = np.array(boot)
    frame_implied = (d.actual_kg / (d.volume_cm3 / 1e6)).to_numpy()
    return {"density_kg_m3": a * 1e6, "loeo_values": np.array(fold),
            "loeo_min": float(np.min(fold)), "loeo_max": float(np.max(fold)),
            "ci95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
            "bootstrap": boot, "frame_implied": frame_implied,
            "config": report.model.as_dict()}


def parameter_sensitivity(report: FitReport) -> dict[str, pd.DataFrame]:
    """Best CV error reachable for each value of each searched parameter."""
    cv = report.cv_table
    out = {}
    for p in ("offset_px", "trim", "slice_count", "calib_method", "calib_scope"):
        g = cv.groupby(p, sort=False).cv_rmse_kg.min().reset_index()
        if p in ("offset_px", "trim"):
            g = g.sort_values(p)
        elif p == "slice_count":
            g["_o"] = g.slice_count.map(lambda v: math.inf if v == "per-column" else v)
            g = g.sort_values("_o").drop(columns="_o")
        out[p] = g.reset_index(drop=True)
    return out


def fitted_records(report: FitReport, records: pd.DataFrame) -> pd.DataFrame:
    """Recompute the records table with the fitted equation.

    ``pred_kg`` is the **out-of-fold** estimate (each event predicted by a
    model that never saw it, so plots show honest accuracy);
    ``pred_final_kg`` uses the final model fitted on all events (use it for new
    data). ``volume_cm3`` and ``pixel_to_cm`` are the fitted settings' values,
    and ``implied_density`` is recomputed from them.
    """
    oof = report.oof_frames[["sample_id", "volume_cm3", "pixel_to_cm", "pred_kg"]]
    drop = [c for c in ("volume_cm3", "pred_kg", "error_kg", "pct_error", "implied_density",
                        "pixel_to_cm") if c in records]
    r = records.drop(columns=drop).merge(
        oof.rename(columns={"volume_cm3": "_v", "pixel_to_cm": "_p", "pred_kg": "_m"}),
        on="sample_id", how="left")
    r["volume_cm3"], r["pixel_to_cm"], r["pred_kg"] = r.pop("_v"), r.pop("_p"), r.pop("_m")
    r["pred_final_kg"] = report.model.predict_from_volume(r.volume_cm3)
    r["error_kg"] = r.pred_kg - r.actual_kg
    r["pct_error"] = 100 * r.error_kg / r.actual_kg
    r["implied_density"] = r.actual_kg / (r.volume_cm3 / 1e6)
    r["status"] = np.where(r.pred_kg.isna() & (r.status == "ok"), "no_fitted_volume", r.status)
    return r
