"""Plots for the fitted mass equation (:mod:`broilerseg.massmodel`)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from broilerseg.massmodel import (
    MassModel,
    adjust_mask,
    parameter_sensitivity,
    trim_ends,
)
from broilerseg.viz.metrics import plot_pred_vs_actual
from broilerseg.viz.style import SERIES, crop_box, image_ax, weight_factor, weight_unit
from broilerseg.volume import align_mask, mask_slice_profile


def _u():
    return weight_factor(), weight_unit()


def plot_mass_fit(report, title: str | None = None, figsize=(17, 5)):
    """Fitted curve, out-of-fold accuracy, and per-event error (axes zoomed to the data)."""
    F, U = _u()
    d, m = report.data, report.model
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for k, (e, g) in enumerate(d.groupby("event_id")):
        ax.scatter(g.volume_cm3, g.actual_kg * F, s=22, color=cmap(k % 10), alpha=0.8,
                   label=f"event {e}" if k < 10 else None)
    vs = np.linspace(d.volume_cm3.min() * 0.85, d.volume_cm3.max() * 1.15, 100)
    ax.plot(vs, m.predict_from_volume(vs) * F, color="black", lw=2,
            label=f"fit: m = a V^{m.b:.3f}")
    a1 = float(np.exp(np.average(np.log(d.actual_kg / d.volume_cm3), weights=d.w)))
    ax.plot(vs, a1 * vs * F, color=SERIES["pred"], ls="--", lw=1.5,
            label=f"b = 1 (density {a1 * 1e6:.0f} kg/m$^3$)")
    # Zoom to the data. Log axes only help when values span a wide range; with birds of
    # similar weight they squash everything into a corner, so use linear axes then.
    ys = np.r_[d.actual_kg.to_numpy(), m.predict_from_volume(vs), a1 * vs] * F
    y_lo, y_hi = np.nanmin(ys), np.nanmax(ys)
    pad = 0.08 * (y_hi - y_lo) if y_hi > y_lo else 0.1 * y_hi
    ax.set_xlim(vs.min(), vs.max())
    ax.set_ylim(max(0, y_lo - pad), y_hi + pad)
    use_log = d.volume_cm3.max() / d.volume_cm3.min() > 3
    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    tag = ", log" if use_log else ""
    ax.set_xlabel(f"adjusted volume V (cm$^3${tag})")
    ax.set_ylabel(f"scale reading ({U}{tag})")
    ax.legend(fontsize=7)
    ax.set_title("Fitted equation (all events)")
    plot_pred_vs_actual(report.oof_events, ax=axes[1], color=SERIES["teacher"],
                        title="Out-of-fold, per event (honest error)")
    ev = report.oof_events.sort_values("actual_kg").reset_index(drop=True)
    pct = 100 * ev.error_kg / ev.actual_kg
    axes[2].bar(range(len(ev)), ev.error_kg * F,
                color=[SERIES["bad"] if v > 0 else SERIES["student"] for v in ev.error_kg])
    for k, v in enumerate(pct):
        axes[2].text(k, ev.error_kg[k] * F, f"{v:+.1f}%", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=7)
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_xticks(range(len(ev)))
    axes[2].set_xticklabels(ev.event_id, rotation=70, fontsize=7)
    axes[2].set_ylabel(f"error ({U})")
    axes[2].set_title("Out-of-fold error per event")
    fig.suptitle(title or m.equation(U), fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_model_comparison(report, title: str = "What each part of the equation buys you"):
    """Leave-one-event-out error of the baseline, density-only and full models."""
    F, U = _u()
    c = report.comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.6 * len(c) + 2.8))
    cols = [SERIES["neutral"], SERIES["pred"], SERIES["teacher"], SERIES["student"]][:len(c)]
    for ax, col, lab in ((axes[0], "rmse_kg", f"RMSE ({U})"), (axes[1], "mape", "MAPE (%)")):
        vals = c[col] * (F if col != "mape" else 1)
        ax.barh(c.model, vals, color=cols)
        for k, v in enumerate(vals):
            ax.text(v, k, f" {v:.1f}" if U == "g" or col == "mape" else f" {v:.3f}",
                    va="center", fontsize=8)
        ax.set_xlabel(lab + " (out-of-fold, event level)")
        ax.invert_yaxis()
    axes[1].set_yticklabels([])
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_parameter_sensitivity(report, title: str = "Best achievable error for each setting"):
    """One panel per searched parameter; the chosen value is highlighted."""
    F, U = _u()
    sens = parameter_sensitivity(report)
    chosen = report.model.as_dict()
    names = {"offset_px": "boundary offset (px)", "trim": "end trim (fraction)",
             "slice_count": "slice count", "calib_method": "calibration method",
             "calib_scope": "calibration scope"}
    fig, axes = plt.subplots(1, len(sens), figsize=(4.2 * len(sens), 4))
    for ax, (p, g) in zip(axes, sens.items(), strict=False):
        x = [str(v) for v in g[p]]
        sel = str(chosen[p])
        colors = [SERIES["teacher"] if v == sel else "#c7c7c7" for v in x]
        ax.bar(x, g.cv_rmse_kg * F, color=colors)
        ax.set_title(names[p])
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_ylim(0, max(g.cv_rmse_kg * F) * 1.15)
    axes[0].set_ylabel(f"CV RMSE ({U})")
    fig.suptitle(title + "  (purple = chosen)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_fold_stability(report,
                        title: str = "Do the fitted numbers change when an event is left out?"):
    """Fitted density (or a) and exponent b for every leave-one-event-out fold."""
    f = report.folds
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    if f.density_kg_m3.notna().all():
        axes[0].plot(f.held_out_event, f.density_kg_m3, "o-", color=SERIES["pred"])
        axes[0].axhline(report.model.a * 1e6, color="black", ls="--", label="all events")
        axes[0].set_ylabel("density (kg/m$^3$)")
    else:
        axes[0].plot(f.held_out_event, f.a, "o-", color=SERIES["pred"])
        axes[0].axhline(report.model.a, color="black", ls="--", label="all events")
        axes[0].set_ylabel("a")
    axes[0].legend()
    axes[1].plot(f.held_out_event, f.b, "o-", color=SERIES["teacher"])
    axes[1].axhline(1, color="gray", ls=":")
    axes[1].set_ylabel("exponent b")
    axes[1].set_title(f"b: {report.b_status()}", fontsize=9)
    for ax in axes:
        ax.set_xlabel("event left out")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_density_estimate(dens: dict, title: str = "The most accurate single density"):
    """Bootstrap distribution, leave-one-event-out values and per-frame implied density."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.3))
    rho, (lo, hi) = dens["density_kg_m3"], dens["ci95"]
    axes[0].hist(dens["bootstrap"], bins=40, color=SERIES["teacher"], alpha=0.75)
    axes[0].axvline(rho, color="black", lw=2, label=f"fit {rho:.0f}")
    axes[0].axvspan(lo, hi, color=SERIES["teacher"], alpha=0.15, label=f"95% CI {lo:.0f}-{hi:.0f}")
    axes[0].set_xlabel("density (kg/m$^3$)")
    axes[0].legend()
    axes[0].set_title("Bootstrap over events")
    v = dens["loeo_values"]
    axes[1].plot(range(len(v)), v, "o-", color=SERIES["pred"])
    axes[1].axhline(rho, color="black", ls="--")
    axes[1].set_xlabel("fold (one event left out)")
    axes[1].set_ylabel("density (kg/m$^3$)")
    axes[1].set_title(f"Leave-one-event-out: {dens['loeo_min']:.0f}-{dens['loeo_max']:.0f}")
    axes[2].hist(dens["frame_implied"], bins=20, color=SERIES["student"], alpha=0.8)
    axes[2].axvline(rho, color="black", lw=2)
    axes[2].set_xlabel("implied density of each frame (kg/m$^3$)")
    axes[2].set_title("Frame-to-frame spread")
    fig.suptitle(f"{title}: {rho:.0f} kg/m$^3$ (95% CI {lo:.0f}-{hi:.0f})", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_mask_adjustments(mask: np.ndarray, model: MassModel,
                          title: str = "What the fitted shape settings do to one bird"):
    """Original mask, boundary offset, end trim and the fitted slices."""
    x1, y1, x2, y2 = crop_box(mask, mask.shape, pad=0.1)
    crop = np.asarray(mask)[y1:y2, x1:x2] > 0
    adj = adjust_mask(crop, model.offset_px)
    aligned, _ = align_mask(adj)
    trimmed = trim_ends(aligned, model.trim)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    image_ax(axes[0], crop, f"Mask ({crop.sum():,} px)")
    diff = np.zeros((*crop.shape, 3), np.uint8)
    diff[crop & adj] = (200, 200, 200)
    diff[adj & ~crop] = (40, 200, 40)
    diff[crop & ~adj] = (220, 40, 40)
    image_ax(axes[1], diff, f"Boundary offset {model.offset_px:+d} px "
                            "(green added, red removed)")
    tri = np.zeros((*aligned.shape, 3), np.uint8)
    a = aligned > 0
    tri[a] = (200, 200, 200)
    tri[a & ~trimmed] = (220, 40, 40)
    image_ax(axes[2], tri, f"End trim {100 * model.trim:.0f}% each end (red = not counted)")
    prof = mask_slice_profile(trimmed, model.slice_count)
    canvas = np.zeros((*trimmed.shape, 3), float)
    colors = [np.array([0.35, 0.65, 0.95]), np.array([0.95, 0.55, 0.25])]
    for k, (xs, xe) in enumerate(zip(prof["x_start"], prof["x_end"], strict=False)):
        sub = trimmed[:, xs:xe + 1]
        canvas[:, xs:xe + 1][sub] = colors[k % 2]
    axes[3].imshow(canvas)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    axes[3].grid(False)
    step = max(1, len(prof["sample_x"]) // 40)
    for k in range(0, len(prof["sample_x"]), step):
        axes[3].plot([prof["sample_x"][k]] * 2, [prof["y_min"][k], prof["y_max"][k]],
                     color="yellow", lw=1)
    axes[3].set_title(f"{len(prof['sample_x'])} slices")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_cv_landscape(report, top: int = 25, title: str = "Best configurations (cross-validated)"):
    """Table-like bar chart of the top configurations."""
    F, U = _u()
    t = report.cv_table.head(top).iloc[::-1]
    labels = [f"off {r.offset_px:+d} | trim {r.trim:.2f} | n {r.slice_count} | "
              f"{r.calib_method} / {r.calib_scope} | {r.b_mode}" for r in t.itertuples()]
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(t) + 1.5))
    ax.barh(range(len(t)), t.cv_rmse_kg * F, color=SERIES["teacher"])
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"CV RMSE ({U})")
    ax.set_title(title)
    fig.tight_layout()
    return fig


__all__ = ["plot_mass_fit", "plot_model_comparison", "plot_parameter_sensitivity",
           "plot_fold_stability", "plot_density_estimate", "plot_mask_adjustments",
           "plot_cv_landscape", "plot_nested_choices"]


def plot_nested_choices(report, title: str = "Settings chosen when each event is held out"):
    """Nested CV: which settings the full search picks without each event, and its error."""
    F, U = _u()
    n = report.nested_folds
    if n.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "nested CV not run (needs >= 3 events)", ha="center")
        ax.axis("off")
        return fig
    if len(n) > 40:
        title += f" (first 40 of {len(n)} events)"
        n = n.head(40)
    params = ["offset_px", "trim", "slice_count", "calib_method", "calib_scope", "b_mode"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 0.45 * len(n) + 2.5),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    cell = [[str(r[p]) for p in params] for _, r in n.iterrows()]
    chosen = report.model.as_dict()
    chosen["b_mode"] = "b = 1" if report.fix_b else "b fitted"
    colors = [["#d9f0d3" if str(r[p]) == str(chosen[p]) else "#fde0dd" for p in params]
              for _, r in n.iterrows()]
    axes[0].axis("off")
    tab = axes[0].table(cellText=cell, colLabels=params, rowLabels=list(n.held_out_event),
                        cellColours=colors, loc="center")
    tab.scale(1, 1.4)
    axes[0].set_title("green = same as the final model, red = different")
    pct = 100 * n.error_kg / n.actual_kg
    axes[1].barh(n.held_out_event.astype(str), n.error_kg * F,
                 color=[SERIES["bad"] if v > 0 else SERIES["student"] for v in n.error_kg])
    for k, v in enumerate(pct):
        axes[1].text(n.error_kg.iloc[k] * F, k, f" {v:+.1f}%", va="center", fontsize=8)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel(f"held-out error ({U})")
    axes[1].set_title("Unbiased error per event")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
