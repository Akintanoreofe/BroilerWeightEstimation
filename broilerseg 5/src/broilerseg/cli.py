"""Command-line interface for long runs outside Jupyter.

Examples
--------
::

    broilerseg label  VIDEOS_DIR  LABELS_DIR --frames-per-video 30
    broilerseg build  LABELS_DIR  YOLO_DIR
    broilerseg train  YOLO_DIR/data.yaml --epochs 100 --imgsz 640
    broilerseg export runs/broilerseg/student/weights/best.pt --imgsz 640
"""

from __future__ import annotations

import argparse


def main(argv=None) -> None:
    """Entry point of the ``broilerseg`` command."""
    p = argparse.ArgumentParser(prog="broilerseg", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("label", help="auto-label video frames with SAM 3")
    a.add_argument("video_dir")
    a.add_argument("out_dir")
    a.add_argument("--frames-per-video", type=int, default=20)
    a.add_argument("--strategy", choices=["uniform", "random", "motion"], default="uniform")
    a.add_argument("--device", default=None)
    a.add_argument("--max-side", type=int, default=1280)

    b = sub.add_parser("build", help="convert SAM 3 labels to a YOLO-seg dataset")
    b.add_argument("label_dir")
    b.add_argument("out_dir")
    b.add_argument("--val", type=float, default=0.2)
    b.add_argument("--min-score", type=float, default=0.0)

    t = sub.add_parser("train", help="train the YOLOv8-seg student")
    t.add_argument("data_yaml")
    t.add_argument("--model", default="yolov8n-seg.pt")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)

    e = sub.add_parser("export", help="export weights for Raspberry Pi (NCNN + ONNX)")
    e.add_argument("weights")
    e.add_argument("--imgsz", type=int, default=640)

    args = p.parse_args(argv)
    if args.cmd == "label":
        from broilerseg.labeling import auto_label_videos
        from broilerseg.segmentation.sam3 import Sam3Segmenter

        seg = Sam3Segmenter(device=args.device, max_side=args.max_side)
        m = auto_label_videos(args.video_dir, args.out_dir, seg, args.frames_per_video,
                              args.strategy)
        print(f"Labelled {len(m)} frames -> {args.out_dir}")
    elif args.cmd == "build":
        from broilerseg.yolo_dataset import build_yolo_dataset

        s = build_yolo_dataset(args.label_dir, args.out_dir, (1 - args.val, args.val, 0.0),
                               min_score=args.min_score)
        print(s.groupby("split").size().to_string())
    elif args.cmd == "train":
        from broilerseg.distill import train_student

        run = train_student(args.data_yaml, args.model, args.epochs, args.imgsz, args.batch)
        print(f"Run folder: {run}")
    elif args.cmd == "export":
        from broilerseg.distill import export_for_raspberry_pi

        for fmt, path in export_for_raspberry_pi(args.weights, args.imgsz).items():
            print(f"{fmt}: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
