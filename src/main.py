"""Camera-agnostic broiler segmentation and outline-volume CLI."""

import argparse
from datetime import datetime
import re
from pathlib import Path
from typing import Optional

import cv2
from BirdSelection import FeederRelativeBirdSelector
from FeederReference import FeederReference
from Factories import SegmentationMethod, SegmenterFactory
from OutputWriters import FeatureCSVWriter
from SegmentationStrategy import ClassicalSegmenter
from SliceVolumeEstimator import (
    Circular,
    DensityWeightModel,
    Elliptical,
    ProfileScaled,
    SliceVolumeEstimator,
)
from defaults import (
    DEBUG_EVERY_N_FRAMES,
    DEFAULT_N_SLICES,
    DENSITY_G_CM3,
    ELLIPTICAL_DEPTH_RATIO,
    VIDEO_EXTENSIONS,
)


def recording_date_from_name(video_name: str) -> str:
    """Parse a date in YYYY-MM-DD or YYYYMMDD form; return blank if absent."""
    match = re.search(r"(?<!\d)(\d{4})[-_]?(\d{2})[-_]?(\d{2})(?!\d)", video_name)
    if not match:
        return ""
    try:
        return datetime(*(int(value) for value in match.groups())).date().isoformat()
    except ValueError:
        return ""


def _cross_section(name: str):
    if name == "circular":
        return Circular()
    if name == "elliptical":
        return Elliptical(ELLIPTICAL_DEPTH_RATIO)
    if name == "profile":
        return ProfileScaled()
    raise ValueError(f"Unknown cross-section model: {name}")


def _debug_overlay(frame, feeder_reference, candidate, result, destination):
    rectified_frame = feeder_reference.rectify(frame)
    rectified_mask = feeder_reference.rectify(candidate.mask, is_mask=True)
    import matplotlib.pyplot as plt
    import visualize

    figure = visualize.plot_slices(
        rectified_frame, rectified_mask, result, feeder_reference.px_per_cm
    )
    try:
        figure.savefig(str(destination), dpi=120)
    finally:
        plt.close(figure)


def process_video(
    video_path: Path,
    output_dir: Path,
    segmenter_name: str = "otsu2",
    n_slices: Optional[int] = DEFAULT_N_SLICES,
    cross_section_name: str = "circular",
    debug_every: int = DEBUG_EVERY_N_FRAMES,
) -> Path:
    """Run all stages for one video and write its feature table."""
    if debug_every < 0:
        raise ValueError("debug_every must be nonnegative")
    reference = FeederReference.from_video(str(video_path))
    segmenter_method = (SegmentationMethod.OTSU2 if segmenter_name == "otsu2"
                        else SegmentationMethod.CLASSICAL)
    segmenter = SegmenterFactory().create(
        segmenter_method,
        feeder_reference=reference,
    ) if segmenter_name == "otsu2" else ClassicalSegmenter()
    candidate_extractor = SegmenterFactory().create(
        SegmentationMethod.OTSU2, feeder_reference=reference
    )
    selector = FeederRelativeBirdSelector(reference)
    estimator = SliceVolumeEstimator(
        n_slices=n_slices,
        cross_section=_cross_section(cross_section_name),
        weight_model=DensityWeightModel(DENSITY_G_CM3),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_features.csv"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_idx = 0
    debug_idx = 0
    try:
        with FeatureCSVWriter(str(output_path)) as writer:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_idx += 1
                segmented = segmenter.segment(frame)
                if segmenter_name == "otsu2":
                    candidates = segmented.candidates
                    occluders = segmented.rejected_elongated
                else:
                    candidates, occluders = candidate_extractor._components(segmented, frame)
                selection = selector.select(candidates, frame, occluders)
                candidate = selection.selected
                if candidate is None:
                    rejected = [item for item in candidates
                                if "dist_feeder_radii" in item.quality]
                    if not rejected:
                        continue
                    candidate = min(
                        rejected,
                        key=lambda item: item.quality["dist_feeder_radii"],
                    )
                rectified_mask = reference.rectify(candidate.mask, is_mask=True)
                volume_result = estimator.estimate(rectified_mask, reference.px_per_cm)
                quality = candidate.quality
                record = {
                    "video_name": video_path.name,
                    "recording_date": recording_date_from_name(video_path.name),
                    "frame_idx": frame_idx,
                    "timestamp_s": (frame_idx - 1) / fps if fps > 0 else "",
                    "segmenter": segmenter_name,
                    "track_id": candidate.track_id if candidate.track_id >= 0 else "",
                    "measurable": candidate.measurable,
                    "feeder_contact": quality.get("feeder_contact", ""),
                    "occluder_contact": quality.get("occluder_contact", ""),
                    "solidity": quality.get("solidity", ""),
                    "ellipse_aspect": quality.get("ellipse_aspect", ""),
                    "sharpness": quality.get("sharpness", ""),
                    "dist_feeder_radii": quality.get("dist_feeder_radii", ""),
                    "area_cm2": volume_result.area_cm2,
                    "length_cm": volume_result.length_cm,
                    "max_width_cm": volume_result.max_width_cm,
                    "perimeter_cm": volume_result.perimeter_cm,
                    "volume_cm3": volume_result.volume_cm3,
                    "n_slices": volume_result.n_slices,
                    "cross_section": volume_result.cross_section,
                    "k": volume_result.k if volume_result.k is not None else "",
                    "px_per_cm": reference.px_per_cm,
                    "feeder_major_px": reference.major_px,
                    "feeder_minor_px": reference.minor_px,
                    "calib_uncertainty_pct": reference.calib_uncertainty_pct,
                    "slice_widths_cm": volume_result.slice_widths_cm,
                    "weight_kg_est": volume_result.weight_kg_est,
                }
                writer.write(record)
                if candidate.measurable and debug_every and debug_idx % debug_every == 0:
                    debug_path = output_dir / f"{video_path.stem}_frame_{frame_idx:06d}.png"
                    _debug_overlay(frame, reference, candidate, volume_result, debug_path)
                    debug_idx += 1
    finally:
        capture.release()
    return output_path


def video_paths(source: Path):
    """Yield supported video files from a file or directory."""
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video file type: {source.suffix}")
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*")
                      if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    raise FileNotFoundError(f"Video path does not exist: {source}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="video file or directory")
    parser.add_argument("--segmenter", choices=("otsu2", "classical"), default="otsu2")
    parser.add_argument("--n-slices", type=int, default=DEFAULT_N_SLICES)
    parser.add_argument("--cross-section", choices=("circular", "elliptical", "profile"),
                        default="circular")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--debug-every", type=int, default=DEBUG_EVERY_N_FRAMES,
                        help="write a PNG every N measurable frames; 0 disables")
    args = parser.parse_args(argv)
    if args.n_slices < 1:
        parser.error("--n-slices must be positive")
    paths = video_paths(args.video)
    if not paths:
        parser.error(f"No supported videos found under {args.video}")
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        parser.error("Video files must have unique stems to avoid overwriting output files")
    for path in paths:
        output = process_video(path, args.out, args.segmenter, args.n_slices,
                               args.cross_section, args.debug_every)
        print(f"[main] Wrote {output}")


if __name__ == "__main__":
    main()
