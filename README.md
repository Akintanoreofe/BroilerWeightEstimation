# Broiler Weight Estimation

> Forked from [AminNasiri63/BroilerWeightEstimation](https://github.com/AminNasiri63/BroilerWeightEstimation)

A production-oriented computer vision pipeline for estimating broiler chicken weight from video streams.
The system is designed for real-world poultry houses, emphasizing robustness, interpretability, and low computational cost.

Unlike deep-learning–heavy solutions, this project focuses on classical CV and geometric modeling, making it suitable for edge deployment, rapid prototyping, and environments with limited labeled data.

---

## Demo Video

https://github.com/user-attachments/assets/fe745700-ba96-4303-a332-41783033921b

---

## Why This Project Matters

- Enables non-contact weight monitoring in poultry farms.
- Reduces reliance on manual weighing.
- Designed for continuous, camera-based operation.
- Prioritizes explainability and engineering simplicity.
- Relevant Domains: AgTech & Precision Livestock · Edge / Applied Computer Vision.
---

## Features

- **End-to-End Video Pipeline**: Per-video feeder calibration, segmentation, object validation, and outline-based volume estimation.
- **Classical, Learning-Free Segmentation**: Two-stage Otsu on Lab whiteness, with the original V-channel segmenter retained as a baseline.
- **Robust Object Selection**: Overlap rejection, boundary-touching instance removal, confidence scoring, and configurable Top-K candidate selection.
- **Pose-Normalized, Density-Aware Weight Estimation**: Orientation normalization + geometry-based volume computation, converted to weight using configurable broiler density (default: 900 kg/m³).
- **Structured Outputs**: CSV logging and annotated video output for debugging and offline analysis.
- **Modular Architecture**: Strategy and Factory patterns enabling extensibility and future DL integration.

## Camera-agnostic pipeline

The current pipeline calibrates each video from the orange feeder (14 in /
35.56 cm), performs two-stage Otsu segmentation in Lab whiteness, rectifies the
feeder ellipse, and estimates outline volume in equal slices. It does not read
camera IDs, `cameras_config.json`, hand-drawn ROIs, or scene masks. The
`classical` segmenter remains available as a baseline.

```bash
python src/main.py --video PATH_OR_DIR --segmenter otsu2 --n-slices 20 \
  --cross-section circular --out results/
```

`--video` accepts one video or a directory of videos; use `--segmenter
classical` to compare against the original V-channel implementation.
Cross-sections are `circular`, `elliptical`, or `profile`. Each video's
`*_features.csv` preserves per-frame measurements and rejected selected-bird
frames, including quality metrics, slice widths (JSON), volume in cm³, and
estimated weight in kg. Calibration requires at least five usable sampled
feeder fits. `src/defaults.py` contains global physical and ratio defaults;
`PLANE_CORRECTION` defaults to 1.0. Install `requirements-visualization.txt`
for the walkthrough notebook and optional slice-figure debug PNGs (enabled
with `--debug-every N`).

---

## Project Structure

```
├── src
    ├── main.py                    # Camera-agnostic CLI
    ├── FeederReference.py         # Multi-frame feeder ellipse calibration
    ├── BirdSelection.py           # Feeder-relative selection and quality gates
    ├── SliceVolumeEstimator.py    # Outline-based volume strategies
    ├── main execution.py          # Legacy entry point
    ├── VideoProcessorPipeline.py  # Core video processing pipeline
    ├── CameraLoader.py           # Camera configuration loader
    ├── SegmentationStrategy.py   # Segmentation implementations (Classical/Deep Learning)
    ├── ObjectFilterStrategy.py   # Object filtering and scoring logic
    ├── AreaCalculatorStrategy.py # Volume/weight calculation strategies
    ├── OutputWriters.py          # Output formatters (CSV)
    ├── Factories.py              # Factory classes for strategy creation
    ├── VideoIOClass.py           # Video input/output utilities
    ├── defaults.py                # Global unit-suffixed defaults
    ├── cameras_config.json       # Optional legacy configuration
    ├── broiler_template.png      # Template mask for broiler matching
    └── cam1.mp4                  # Sample video file
├── requirements.txt
└── README.md
```
---

## Installation

### Prerequisites

- Python 3.9+
- OpenCV 4.x
- NumPy

### Dependencies

```bash
pip install opencv-python numpy
```
---

## Legacy configuration

The older `main execution.py` and validation runner accept camera/polygon
configuration for backwards compatibility. The camera-agnostic `src/main.py`
above does not consume `cameras_config.json`. Legacy camera entries may define:

- `crop`: Region of interest coordinates (x1, y1, x2, y2).
- `feeder`: Feeder area coordinates (for calibration).
- `pixel_to_cm`: Pixel-to-centimeter calibration factor.

Example configuration:

```json
{
  "cameras": {
    "cam1": {
      "crop": {"x1": 1900, "y1": 590, "x2": 3180, "y2": 1475},
      "feeder": {"x1": 1700, "y1": 700, "x2": 2250, "y2": 1100},
      "pixel_to_cm": 0.09
    }
  }
}
```
### Legacy camera calibration

- The legacy runner detects the feeder in the first frame and uses an enclosing circle. The current `FeederReference` instead fits ellipses across sampled video frames and reports calibration uncertainty.

---

## Legacy method overview
The entry point calibrates the video from its first frame, then processes each frame independently. It creates a binary foreground mask, selects likely single-broiler contours, and estimates each selected bird's mass from calibrated circular slices through its mask. The diagram shows the implemented classical-segmentation path:

```mermaid
flowchart TD
    A(["Start"]) --> B["Open configured video"]
    B --> C["Read first frame and detect orange feeder"]
    C --> D{"Feeder detected<br>and measurable?"}
    D -- No --> E["Stop with calibration error"]
    D -- Yes --> F["Calculate centimeters per pixel<br>from known 35.56 cm feeder diameter"]
    F --> G["Read next video frame"]
    G --> H{"Frame available<br>and within limit?"}
    H -- No --> Z["Release resources and finish"]
    H -- Yes --> I["Convert frame to HSV<br>and blur brightness channel"]
    I --> J["Otsu threshold brightness<br>and remove orange feeder pixels"]
    J --> K["Clean mask with morphology"]
    K --> L["Find external contours"]
    L --> C1["Process next external contour"]
    C1 --> M{"Contour passes area,<br>size, and border checks?"}
    M -- No --> Y{"More contours?"}
    M -- Yes --> N["Distance transform"]
    N --> O{"Exactly one peak?"}
    O -- No --> Y
    O -- Yes --> P["Rotate and align mask"]
    P --> Q["Compare normal and flipped masks<br>with broiler template"]
    Q --> X["Store candidate scores"]
    X --> Y
    Y -- Yes --> C1
    Y -- No --> R["Combine scores and keep<br>top-K candidates above threshold"]
    R --> S["For each selected candidate:<br>measure mask height per image column"]
    S --> T["Model each column as a circular slice<br>using pixel-to-centimeter scale"]
    T --> U["Sum slice volumes and multiply<br>by assumed density (900 kg/m³)"]
    U --> V["Write estimate to CSV<br>and annotate frame"]
    V --> W["Write annotated video frame"]
    W --> G
```

Volume estimation approximates each occupied image column as a cylinder slice: the mask's vertical span gives the slice diameter, and the calibrated pixel scale gives its dimensions in centimeters. Summing slice volumes and multiplying by the assumed broiler density produces the estimated mass in kilograms.


<img width="224" height="224" alt="broiler_sliced" src="https://github.com/user-attachments/assets/54cb91df-7abb-4148-bed2-62cd404cd057" />


---

## Legacy usage

1. Configure camera parameters in `cameras_config.json`.
2. Place input video files in the project directory.
3. (Optional) Adjust the Top-K candidate selection via the `top_k` parameter in the object filtering stage.
4. Update the video path in `main_execution.py` if needed.
5. Run the main script:

```bash
python "main execution.py"
```

The system processes the input video and saves an annotated output video along with a camera-specific CSV file named `<camera_id>_results.csv`.

### Output Format

The CSV output contains the following columns:

| Field        | Description |
|-------------|-------------|
| `camera_id` | Camera identifier |
| `frame_idx` | Frame number |
| `label`     | Object label (default: `"broiler"`) |
| `score`     | Combined confidence score |
| `volume`    | Estimated weight (kg) |
| `crop_region` | Bounding box coordinates `(x, y, w, h)` |

### Validate Weight Estimates

#### FlockLabel export and notebook

In a remote VS Code workspace, install the pipeline and notebook dependencies
in the Python environment used by the notebook kernel:

```bash
python -m pip install -r requirements.txt ipykernel
```

Open `src/validation_walkthrough.ipynb` in VS Code and select that Python
kernel. Copy the FlockLabel export into a `dataset` directory at the
repository root, or set `EXPORT_DIR` in the first code cell to the export
directory, then run all cells.
Relative export paths are resolved from the repository root. The root is
automatically found when the kernel's working directory is inside the
repository. Otherwise, set `REPO_ROOT` to the absolute remote path of the
repository folder (containing `src/run_eval.py`), not to the script itself. The
notebook displays a sample frame, calibration, segmentation mask, filtering and
weight estimates, then processes every clip, reports validation metrics and
generates comparison videos. The videos show ground-truth boxes in green and
estimated boxes in red, including unmatched boxes. The first labeled clip is
embedded in the notebook; all comparisons are saved under
`<export>/results/comparisons/`. Some browsers may not support OpenCV's MP4
codec; open the saved video locally if necessary.
For each selected bird in the preview frame, the notebook shows its unrotated
binary mask, that mask overlaid on the original frame, and the aligned mask
used to calculate its weight. Change `PREVIEW_VIDEO` or `PREVIEW_FRAME` to
inspect a different frame; no bird masks are shown when none passes selection.
The slice visualization colors the aligned mask by column group and shows
the sampled diameters, individual slice volumes, and their total. These
circular sections are model assumptions, not measured 3D anatomy.

The notebook assumes the clips are already cropped and does not require a
weighing-platform boundary. It uses the full detected silhouette for mass
estimation. Cropping does not establish that a bird is weight-bearing on the
scale; occlusion, multiple birds, edge clipping and segmentation errors can
invalidate the association with a reading. Review the frames and matches.
The filter preview reports why contours were rejected before scoring.
Frame-edge contours are allowed in the notebook by default, assuming the
dataset contains complete silhouettes. Set `ALLOW_FRAME_EDGE = False` to
exclude them if birds can be clipped. Multiple distance-transform peaks can
indicate touching birds, but may also arise from irregular segmentation. The report does not count
candidates excluded by score or top-K selection.

The optional, disabled-by-default tuning cell tries slice counts and densities
on matched, labeled scale frames. Supply disjoint calibration and held-out
event IDs; it ranks by calibration event-level RMSE and reports the held-out
score only for the selected grid candidate. Set the selected `SLICE_COUNT`
and `DENSITY` in the setup cell and rerun the pipeline to apply them. Tuning
and evaluation on the same events is not an independent accuracy assessment.
The model assumes circular cross-sections inferred from a top-view silhouette,
not measured depth. `SLICE_COUNT = None` retains the original one-column-per-
slice calculation; otherwise each group of columns uses its middle column's
diameter and the group's width.

The expected export structure is:

```text
export/
├── test_videos/
│   ├── metadata.csv    # video_name
│   └── <video files listed in metadata>
└── ground_truth/
    ├── events.csv      # video_name, optional pixel_to_cm
    └── gt_boxes.csv    # video_name, frame_idx, x, y, w, h, weight_kg, event_id
```

Boxes use pixel `(x, y, width, height)` coordinates. The pipeline numbers
frames starting at 1; set `GT_FRAME_OFFSET = 1` for zero-based ground-truth
frame numbers, or leave it at 0 for one-based numbers. Video names must have
unique stems. `events.csv` can supply a positive `pixel_to_cm` per video;
otherwise calibration requires a detectable orange feeder in the first frame.
Its 14-inch (35.56 cm) diameter divided by its measured pixel diameter gives
centimeters per pixel. The complete feeder must remain visible after cropping
and other orange objects can distort detection; if it is absent or not
reliably measurable, supply a valid `pixel_to_cm` measured for the exported video. A crop
does not change pixel size, but resizing does.
The runner reports calibration failures and exits unsuccessfully if any clip
was skipped. Existing result files are overwritten.

The same processing and evaluation can be run without Jupyter:

```bash
python src/run_eval.py /path/to/export
python src/evaluate_against_gt.py /path/to/export/ground_truth/gt_boxes.csv \
  --results-dir /path/to/export/results --out /path/to/export/results/evaluation \
  --gt-frame-offset 0
```

The CLI retains its original all-bird behavior unless `--scale-polygons
/path/to/polygons.json` is supplied. The JSON maps each exact video name to
its polygon points, for example `{"clip.mp4": [[30, 45], [420, 40],
[430, 430], [35, 430]]}`. Use
`--min-scale-overlap` to configure the optional filter, `--allow-frame-edge`
to match the notebook's default edge policy, and `--density` and
`--slices` to match notebook settings; pass the same density to the evaluator
with `--density`.

The evaluator greedily matches boxes by IoU on the same video and frame.
Estimates on frames without ground-truth labels are counted as unmatched, not
necessarily confirmed false positives. Event weight metrics use the median
of matched scale readings and the median estimate. Fit any suggested density
on separate calibration events and report accuracy on held-out events.

#### CSV weight validation

To compare estimates with measured weights, add an `actual_weight_kg` column to
the results CSV, with one measured weight in kilograms on each corresponding
row. The pipeline's estimated weights are in the `volume` column. Then run:

```bash
python src/weight_validation.py results/cam1_results.csv
```

The script reports mean absolute error (MAE), root mean squared error (RMSE),
mean bias (estimate minus actual), and mean absolute percentage error (MAPE).
Rows with zero actual weight are excluded from MAPE. For different column
names, use `--actual-column` and `--estimated-column`.

---

## Known Tradeoffs
**These are deliberate design choices aimed at reliability, transparency, and low deployment cost.**

- **Simplified Body Model:** The volume strategies approximate 3D cross-sections from a 2D silhouette; validate these assumptions against scale weights.
- **Lighting Sensitivity:** Classical segmentation performs best under controlled illumination, which is a common constraint in commercial poultry houses.
- **Plane Correction:** The feeder rim and bird outline can be at different heights; the global `PLANE_CORRECTION` defaults to 1.0 and requires measured weights for tuning.
- **Template Dependency:** Shape-based validation relies on orientation-consistent templates, prioritizing precision over recall.

---

## Practical Notes

- Continuous per-frame estimation is often unnecessary; sampling frames and top-ranked broilers within a fixed time window can reduce computation without sacrificing monitoring quality.

---

## Roadmap

- Integration of deep learning models for improved segmentation.
- Posture-aware frame selection to improve volume-based weight estimation accuracy.
- Multi-species support with configurable templates.
- Web interface for configuration and monitoring.
- Statistical analysis and reporting features.
---

## License
MIT License.

---

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
