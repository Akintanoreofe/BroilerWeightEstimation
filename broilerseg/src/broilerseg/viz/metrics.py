"""Plots of weight-estimation accuracy and parameter sweeps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from broilerseg.evaluation import event_table, regression_metrics
from broilerseg.viz.style import SERIES, new_ax, to_display, weight_factor, weight_unit


def _metrics_text(m: dict) -> str:
    """Metrics box; ``m`` must already be in the display unit."""
    u, d = weight_unit(), 3 if weight_unit() == "kg" else 0
    return (f"n={m['n']}\nMAE {m['mae']:.{d}f} {u}\nRMSE {m['rmse']:.{d}f} {u}\n"
            f"bias {m['bias']:+.{d}f} {u}\nMAPE {m['mape']:.1f}%\nR$^2$ {m['r2']:.2f}")


def plot_pred_vs_actual(df: pd.DataFrame, pred: str = "pred_kg", actual: str = "actual_kg",
                        ax=None, title: str = "Estimated vs scale weight", band_pct: float = 10,
                        color=None, label: str | None = None, annotate: bool = True,
                        xlabel: str | None = None, ylabel: str | None = None):
    """Scatter with the 1:1 line, a +/- ``band_pct`` % band and the metrics.

    ``df`` holds kilograms; values are shown in the unit set by
    :func:`broilerseg.viz.style.set_weight_unit`.
    """
    fig, ax = new_ax(ax, (6, 6))
    u = weight_unit()
    xlabel = xlabel or f"scale reading ({u})"
    ylabel = ylabel or f"estimated ({u})"
    d = to_display(df, (pred, actual))
    d = d[np.isfinite(d[pred]) & np.isfinite(d[actual])]
    lo = 0.9 * np.nanmin([d[pred].min(), d[actual].min()]) if len(d) else 0
    hi = 1.1 * np.nanmax([d[pred].max(), d[actual].max()]) if len(d) else 1
    xs = np.linspace(lo, hi, 50)
    ax.fill_between(xs, xs * (1 - band_pct / 100), xs * (1 + band_pct / 100), color="gray",
                    alpha=0.12, label=f"+/-{band_pct:.0f}%")
    ax.plot(xs, xs, color="gray", ls="--", lw=1, label="perfect")
    ax.scatter(d[actual], d[pred], s=22, alpha=0.7, color=color or SERIES["pred"],
               edgecolor="white", lw=0.4, label=label)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if annotate:
        ax.text(0.03, 0.97, _metrics_text(regression_metrics(d[actual], d[pred])),
                transform=ax.transAxes, va="top", fontsize=8,
                bbox={"fc": "white", "alpha": 0.85, "ec": "none"})
    ax.legend(loc="lower right")
    return fig


def plot_accuracy_overview(records: pd.DataFrame, title: str = "Weight accuracy"):
    """Frame-level and event-level predicted vs actual, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_pred_vs_actual(records, ax=axes[0], title="Every frame")
    ev = event_table(records)
    plot_pred_vs_actual(ev, ax=axes[1], title="Per event (medians)", color=SERIES["teacher"])
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_error_analysis(records: pd.DataFrame, title: str = "Where do the errors come from?"):
    """Residual histogram, residual vs weight, Bland-Altman and abs-% error CDF."""
    u = weight_unit()
    d = to_display(records).dropna(subset=["pred_kg", "actual_kg"])
    e = d.pred_kg - d.actual_kg
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
    axes[0].hist(e, bins=25, color=SERIES["pred"], edgecolor="white")
    axes[0].axvline(0, color="k", lw=1)
    axes[0].axvline(e.mean(), color=SERIES["bad"], ls="--", label=f"mean {e.mean():+.3g} {u}")
    axes[0].set_xlabel(f"error ({u})")
    axes[0].legend()
    axes[0].set_title("Error distribution")
    axes[1].scatter(d.actual_kg, e, s=14, alpha=0.6, color=SERIES["pred"])
    axes[1].axhline(0, color="k", lw=1)
    if len(d) > 2:
        k, b = np.polyfit(d.actual_kg, e, 1)
        xs = np.linspace(d.actual_kg.min(), d.actual_kg.max(), 10)
        axes[1].plot(xs, k * xs + b, color=SERIES["bad"], label=f"trend {k:+.2f} {u}/{u}")
        axes[1].legend()
    axes[1].set_xlabel(f"scale reading ({u})")
    axes[1].set_ylabel(f"error ({u})")
    axes[1].set_title("Error vs bird weight")
    mean = (d.pred_kg + d.actual_kg) / 2
    sd = e.std()
    axes[2].scatter(mean, e, s=14, alpha=0.6, color=SERIES["teacher"])
    for v, ls in ((e.mean(), "-"), (e.mean() + 1.96 * sd, "--"), (e.mean() - 1.96 * sd, "--")):
        axes[2].axhline(v, color="gray", ls=ls)
    axes[2].set_xlabel(f"mean of estimate and scale ({u})")
    axes[2].set_ylabel(f"difference ({u})")
    axes[2].set_title("Bland-Altman (95% limits of agreement)")
    ape = np.sort(np.abs(e / d.actual_kg) * 100)
    axes[3].plot(ape, np.linspace(0, 100, len(ape)), color=SERIES["student"], lw=2)
    for t in (5, 10, 20):
        frac = (ape <= t).mean() * 100 if len(ape) else 0
        axes[3].axvline(t, color="gray", ls=":")
        axes[3].text(t, 5, f"{frac:.0f}% <= {t}%", rotation=90, fontsize=8, va="bottom")
    axes[3].set_xlabel("absolute % error")
    axes[3].set_ylabel("% of frames")
    axes[3].set_title("Cumulative error")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_per_event(records: pd.DataFrame, title: str = "Every weighing event",
                   max_events: int = 60):
    """Median estimate vs median scale per event, frame estimates as dots (<= ``max_events``)."""
    records = to_display(records)
    ev = event_table(records).sort_values("actual_kg").reset_index(drop=True)
    if len(ev) > max_events:
        keep = np.unique(np.linspace(0, len(ev) - 1, max_events).astype(int))
        title += f" (showing {len(keep)} of {len(ev)}, spread over the weight range)"
        ev = ev.iloc[keep].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(ev) + 3), 4.8))
    x = np.arange(len(ev))
    ax.bar(x - 0.2, ev.actual_kg, 0.4, color=SERIES["actual"], label="scale (median)")
    ax.bar(x + 0.2, ev.pred_kg, 0.4, color=SERIES["pred"], label="estimate (median)")
    rng = np.random.default_rng(0)
    for k, e in enumerate(ev.event_id):
        y = records.loc[(records.event_id == e), "pred_kg"].dropna()
        ax.scatter(k + 0.2 + rng.uniform(-0.12, 0.12, len(y)), y, s=8, color="black", alpha=0.5)
        ax.text(k, max(ev.actual_kg[k], ev.pred_kg[k]), f"{ev.pct_error[k]:+.0f}%",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(ev.event_id, rotation=70, fontsize=7)
    ax.set_ylabel(weight_unit())
    ax.legend()
    ax.set_title(f"{title} (dots = individual frames, label = % error)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_frame_traces(records: pd.DataFrame, max_events: int = 12, ncols: int = 4,
                      title: str = "Estimate frame by frame within each event"):
    """Small multiples: estimate and scale reading along each clip."""
    records = to_display(records)
    events = list(dict.fromkeys(records.event_id))[:max_events]
    nrows = int(np.ceil(len(events) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows), squeeze=False,
                             sharey=True)
    for ax, e in zip(axes.ravel(), events, strict=False):
        d = records[records.event_id == e].sort_values("frame_label")
        ax.plot(d.frame_label, d.actual_kg, color=SERIES["actual"], lw=2, label="scale")
        ax.plot(d.frame_label, d.pred_kg, marker="o", ms=3, color=SERIES["pred"],
                label="estimate")
        bad = d[d.pred_kg.isna()]
        ax.scatter(bad.frame_label, np.full(len(bad), np.nanmin(d.actual_kg) * 0.9),
                   marker="x", color=SERIES["bad"], label="no estimate")
        ax.set_title(str(e), fontsize=9)
        ax.set_xlabel("frame")
    for ax in axes.ravel()[len(events):]:
        ax.axis("off")
    axes[0, 0].set_ylabel(weight_unit())
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_status_breakdown(records: pd.DataFrame, ax=None,
                          title: str = "Why some frames have no estimate"):
    """Bar chart of the pipeline outcome per frame."""
    fig, ax = new_ax(ax, (7, 3.5))
    status = records.status.where(records.pred_kg.notna() | (records.status != "ok"),
                                  "no_volume")
    c = status.value_counts()
    ax.barh(c.index, c.values, color=[SERIES["good"] if s == "ok" else SERIES["bad"]
                                      for s in c.index])
    for k, v in enumerate(c.values):
        ax.text(v, k, f" {v} ({100 * v / len(records):.0f}%)", va="center", fontsize=8)
    ax.set_xlabel("frames")
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_implied_density(records: pd.DataFrame, reference: float = 900.0,
                         title: str = "What density would make each frame exact?"):
    """Histogram of actual / volume and its relation to bird size."""
    d = records.dropna(subset=["implied_density"])
    d = d[np.isfinite(d.implied_density)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.3))
    axes[0].hist(d.implied_density, bins=30, color=SERIES["teacher"], edgecolor="white")
    med = d.implied_density.median() if len(d) else np.nan
    axes[0].axvline(med, color="black", lw=2, label=f"median {med:.0f}")
    axes[0].axvline(reference, color=SERIES["bad"], ls="--", label=f"repo default {reference:.0f}")
    axes[0].set_xlabel("implied density (kg/m$^3$)")
    axes[0].legend()
    axes[0].set_title("Distribution")
    axes[1].scatter(d.area_cm2, d.implied_density, s=14, alpha=0.6, color=SERIES["teacher"])
    axes[1].set_xlabel("silhouette area (cm$^2$)")
    axes[1].set_ylabel("implied density")
    axes[1].set_title("vs bird size (a slope = size bias)")
    axes[2].scatter(d.actual_kg * weight_factor(), d.implied_density, s=14, alpha=0.6,
                    color=SERIES["student"])
    axes[2].set_xlabel(f"scale reading ({weight_unit()})")
    axes[2].set_title("vs weight")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_feature_relationships(records: pd.DataFrame,
                               title: str = "Which mask measurement tracks weight best?"):
    """Scale reading against area, length, width and volume, with correlation."""
    feats = [("area_cm2", "area (cm$^2$)"), ("length_cm", "length (cm)"),
             ("max_width_cm", "max width (cm)"), ("volume_cm3", "volume (cm$^3$)")]
    d = to_display(records).dropna(subset=["pred_kg"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), sharey=True)
    for ax, (c, lab) in zip(axes, feats, strict=False):
        ax.scatter(d[c], d.actual_kg, s=14, alpha=0.6, color=SERIES["pred"])
        r = np.corrcoef(d[c], d.actual_kg)[0, 1] if len(d) > 2 else np.nan
        ax.set_xlabel(lab)
        ax.set_title(f"r = {r:.2f}")
    axes[0].set_ylabel(f"scale reading ({weight_unit()})")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _sweep_display(grid: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str]:
    """Convert a ``*_kg`` error column of a sweep grid to the display unit."""
    if not metric.endswith("_kg") or weight_unit() == "kg":
        return grid, metric
    new = metric[:-3] + "_" + weight_unit()
    return grid.assign(**{new: grid[metric] * weight_factor()}), new


def plot_sweep_heatmap(grid: pd.DataFrame, metric: str = "held_rmse_kg", ax=None,
                       title: str | None = None):
    """Heatmap of an error metric over slice count x density, best cell outlined."""
    fig, ax = new_ax(ax, (11, 4.8))
    grid, metric = _sweep_display(grid, metric)
    piv = grid.pivot_table(index="slice_count", columns="density_kg_m3", values=metric,
                           sort=False)
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{c:.0f}" for c in piv.columns], rotation=60, fontsize=7)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("density (kg/m$^3$)")
    ax.set_ylabel("slice count")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=metric)
    vals = piv.to_numpy()
    if np.isfinite(vals).any():
        i, j = np.unravel_index(np.nanargmin(vals), vals.shape)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=2.5))
        dec = 1 if metric.endswith("_g") else 3
        ax.set_title(title or f"{metric}: best {vals[i, j]:.{dec}f} at slices={piv.index[i]}, "
                     f"density={piv.columns[j]:.0f}")
    return fig


def plot_sweep_curves(grid: pd.DataFrame, metric: str = "held_rmse_kg", ax=None,
                      title: str = "Error vs density for each slice count"):
    """One line per slice count across densities."""
    fig, ax = new_ax(ax, (8, 4.5))
    grid, metric = _sweep_display(grid, metric)
    cmap = plt.get_cmap("viridis")
    groups = list(grid.groupby("slice_count", sort=False))
    for k, (sc, g) in enumerate(groups):
        ax.plot(g.density_kg_m3, g[metric], marker=".", color=cmap(k / max(1, len(groups) - 1)),
                label=str(sc))
    ax.set_xlabel("density (kg/m$^3$)")
    ax.set_ylabel(metric)
    ax.legend(title="slices", ncol=2, fontsize=7)
    ax.set_title(title)
    return fig


def plot_method_comparison(runs: dict[str, pd.DataFrame],
                           title: str = "SAM 3 teacher vs YOLO student: weight accuracy"):
    """Frame-level metrics bars plus overlaid predicted-vs-actual for each method."""
    names = list(runs)
    f = weight_factor()
    mets = {n: regression_metrics(r.actual_kg * f, r.pred_kg * f) for n, r in runs.items()}
    cov = {n: 100 * r.pred_kg.notna().mean() for n, r in runs.items()}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    cmap = plt.get_cmap("Set2")
    x = np.arange(3)
    w = 0.8 / max(1, len(names))
    for k, n in enumerate(names):
        vals = [mets[n]["mae"], mets[n]["rmse"], abs(mets[n]["bias"])]
        axes[0].bar(x + k * w, vals, w, color=cmap(k), label=n)
    axes[0].set_xticks(x + w * (len(names) - 1) / 2)
    axes[0].set_xticklabels(["MAE", "RMSE", "|bias|"])
    axes[0].set_ylabel(weight_unit())
    axes[0].legend()
    axes[0].set_title("Error (lower is better)")
    colors = [cmap(k) for k in range(len(names))]
    axes[1].bar(names, [mets[n]["mape"] for n in names], color=colors)
    for k, n in enumerate(names):
        axes[1].text(k, mets[n]["mape"], f"{mets[n]['mape']:.1f}%\ncoverage {cov[n]:.0f}%",
                     ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_title("Percentage error and % frames estimated")
    for k, n in enumerate(names):
        plot_pred_vs_actual(runs[n], ax=axes[2], color=cmap(k), label=n, annotate=False,
                            title="Every frame")
    handles, labels = axes[2].get_legend_handles_labels()
    seen = dict(zip(labels, handles, strict=False))         # one entry per label
    axes[2].legend(seen.values(), seen.keys(), loc="lower right")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
