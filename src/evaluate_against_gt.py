"""Match pipeline estimates to FlockLabel ground-truth boxes and score them.

Reads the pipeline's ``<camera_id>_results.csv`` files and the exported
``ground_truth/gt_boxes.csv``, pairs each estimate with the ground-truth box
in the same video and frame by IoU, and writes:

* ``matched.csv``   - one row per matched estimate, with ``actual_weight_kg``
  added so ``weight_validation.py`` can score it unchanged;
* ``per_event.csv`` - one row per video/event (median estimate vs median scale reading);
* a printed summary of frame-level and event-level error and detection rate.
"""
import argparse
import ast
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def iou(a, b):
    """IoU of two (x, y, w, h) boxes."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def load_gt(path, frame_offset=0):
    gt = defaultdict(list)          # (video, frame) -> [row]
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r["box"] = tuple(float(r[k]) for k in ("x", "y", "w", "h"))
            r["weight_kg"] = float(r["weight_kg"])
            gt[(r["video_name"], int(r["frame_idx"]) + frame_offset)].append(r)
    return gt


def load_estimates(results_csv, video_name):
    rows = []
    if not Path(results_csv).is_file():
        print(f"[warn] no results file for {video_name}: {results_csv}")
        return rows
    with open(results_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            r["video_name"] = video_name
            r["frame_idx"] = int(r["frame_idx"])
            r["box"] = tuple(float(v) for v in ast.literal_eval(r["crop_region"]))
            r["volume"] = float(r["volume"])
            rows.append(r)
    return rows


def metrics(pairs):
    err = [e - a for a, e in pairs]
    return {
        "n": len(pairs),
        "mae": statistics.fmean(abs(x) for x in err),
        "rmse": math.sqrt(statistics.fmean(x * x for x in err)),
        "bias": statistics.fmean(err),
        "mape": statistics.fmean([abs(e - a) / abs(a) * 100 for a, e in pairs if a])
        if any(a for a, _ in pairs) else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("gt_csv", type=Path)
    p.add_argument("results", nargs="*",
                   help="video_name=path/to/<camera_id>_results.csv pairs")
    p.add_argument("--results-dir", type=Path,
                   help="folder of <clip stem>_results.csv files, paired automatically")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--gt-frame-offset", type=int, default=0,
                   help="add to GT frame numbers (use 1 for zero-based GT)")
    p.add_argument("--out", type=Path, default=Path("eval_out"))
    p.add_argument("--density", type=float, default=900.0,
                   help="density the pipeline ran with (kg/m3)")
    a = p.parse_args()

    if not 0 <= a.iou <= 1:
        p.error("--iou must be between 0 and 1")
    gt = load_gt(a.gt_csv, a.gt_frame_offset)
    specs = list(a.results)
    if a.results_dir:
        for video in sorted({v for v, _ in gt}):
            specs.append(f"{video}={a.results_dir / (Path(video).stem + '_results.csv')}")
    if not specs:
        p.error("give VIDEO=RESULTS pairs or --results-dir")
    estimates = []
    for spec in specs:
        if "=" not in spec:
            p.error(f"invalid result pair: {spec!r}")
        video, path = spec.split("=", 1)
        estimates += load_estimates(path, video)

    matched, used, n_fp = [], set(), 0
    by_frame = defaultdict(list)
    for e in estimates:
        by_frame[(e["video_name"], e["frame_idx"])].append(e)

    for key, ests in by_frame.items():
        # greedy one-to-one matching, best IoU first
        cands = sorted(((iou(e["box"], g["box"]), i, j)
                        for i, e in enumerate(ests)
                        for j, g in enumerate(gt.get(key, []))), reverse=True)
        taken_e, taken_g = set(), set()
        for score, i, j in cands:
            if score < a.iou or i in taken_e or j in taken_g:
                continue
            taken_e.add(i); taken_g.add(j)
            g, e = gt[key][j], ests[i]
            used.add((key, j))
            matched.append({**{k: e[k] for k in ("video_name", "frame_idx", "volume", "score")},
                            "crop_region": e["crop_region"], "iou": round(score, 3),
                            "actual_weight_kg": g["weight_kg"], "event_id": g["event_id"]})
        n_fp += len(ests) - len(taken_e)

    n_gt = sum(len(v) for v in gt.values())
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "matched.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video_name", "frame_idx", "volume", "score",
                                           "crop_region", "iou", "actual_weight_kg", "event_id"])
        w.writeheader()
        w.writerows(matched)

    events = defaultdict(list)
    actuals = defaultdict(list)
    for m in matched:
        events[m["event_id"]].append(m["volume"])
        actuals[m["event_id"]].append(m["actual_weight_kg"])
    weight_of = {k: statistics.median(v) for k, v in actuals.items()}
    gt_events = {g["event_id"] for rows in gt.values() for g in rows}
    with open(a.out / "per_event.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event_id", "actual_weight_kg", "median_estimate_kg",
                    "n_frames_matched", "iqr_kg"])
        for ev, vals in sorted(events.items()):
            q = statistics.quantiles(vals, n=4) if len(vals) > 1 else [vals[0]] * 3
            w.writerow([ev, weight_of[ev], round(statistics.median(vals), 4),
                        len(vals), round(q[2] - q[0], 4)])

    print(f"GT boxes: {n_gt}  estimates: {len(estimates)}")
    print(f"Matched: {len(matched)}  detection rate: {len(used) / max(n_gt, 1):.1%}  "
          f"unmatched estimates: {n_fp}")
    if matched:
        f = metrics([(m["actual_weight_kg"], m["volume"]) for m in matched])
        print(f"Frame level  n={f['n']}  MAE={f['mae']:.3f} kg  RMSE={f['rmse']:.3f} kg  "
              f"bias={f['bias']:+.3f} kg  MAPE={f['mape']:.1f}%")
        ev = metrics([(weight_of[k], statistics.median(v)) for k, v in events.items()])
        print(f"Event level  n={ev['n']}/{len(gt_events)}  MAE={ev['mae']:.3f} kg  "
              f"RMSE={ev['rmse']:.3f} kg  bias={ev['bias']:+.3f} kg  MAPE={ev['mape']:.1f}%")
        # Estimates scale linearly with density, so the best density is closed-form.
        act = [weight_of[k] for k in events]
        est = [statistics.median(v) for v in events.values()]
        if sum(y * y for y in est) and sum(est):
            rho_ls = a.density * sum(x * y for x, y in zip(act, est)) / sum(y * y for y in est)
            rho_unbiased = a.density * sum(act) / sum(est)
            print(f"Density fit (event level, from {a.density:g} kg/m3): least-squares "
                  f"{rho_ls:.0f} kg/m3, zero-bias {rho_unbiased:.0f} kg/m3. "
                  "Fit on a calibration subset and report error on held-out events.")
        else:
            print("Density fit unavailable: estimates sum to zero.")


if __name__ == "__main__":
    main()
