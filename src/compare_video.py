"""Render ground-truth and estimated boxes on the original video."""

from collections import defaultdict
from pathlib import Path

import cv2

from evaluate_against_gt import load_estimates, load_gt


def write_comparison(video_path, gt_csv, results_csv, output_path, frame_offset=0, video_name=None):
    """Write all frames with GT in green and estimates in red.

    Frame numbers in the pipeline are one-based; ``frame_offset`` converts
    annotation frame numbers to that convention.
    """
    video_path, output_path = Path(video_path), Path(output_path)
    video_name = video_name if video_name is not None else video_path.name
    if not Path(results_csv).is_file():
        raise FileNotFoundError(f"Pipeline results not found: {results_csv}")
    gt = load_gt(gt_csv, frame_offset)
    gt_by_frame = {frame: rows for (video, frame), rows in gt.items()
                   if video == video_name}
    estimates = defaultdict(list)
    for row in load_estimates(results_csv, video_name):
        estimates[row["frame_idx"]].append(row)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    writer = None
    frame_idx = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video metadata: {video_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            for rows, color, label in (
                (gt_by_frame.get(frame_idx, []), (0, 255, 0), "GT"),
                (estimates.get(frame_idx, []), (0, 0, 255), "Estimate"),
            ):
                for row in rows:
                    x, y, w, h = (int(round(v)) for v in row["box"])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = f"{label}: {row['weight_kg'] if label == 'GT' else row['volume']:.2f} kg"
                    cv2.putText(frame, text, (x, max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Frame {frame_idx} | GT green | Estimate red",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if frame_idx == 0:
        raise RuntimeError(f"No readable frames in {video_path}")
    outside = [i for i in gt_by_frame if i < 1 or i > frame_idx]
    if outside:
        print(f"[warn] {len(outside)} GT frame numbers are outside {video_path.name} "
              f"(valid range 1..{frame_idx}); check frame offset.")
    return output_path
