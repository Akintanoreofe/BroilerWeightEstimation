# broilerseg

**Teacher mask labelling (Grounded SAM 2 or SAM 3) -> YOLOv8-seg knowledge distillation -> slice-based broiler weight estimation.**

`broilerseg` is an experimental, installable Python package with two Jupyter notebooks. It:

1. uses a text-prompted **teacher** model to segment three things in poultry-house footage: the **broiler**, the **scale** and the **feeder (feeding pan)** on the scale. The teacher is either **Grounded SAM 2** (Grounding DINO + SAM 2.1; public, no approval) or **SAM 3** (gated);
2. turns the feeder (known diameter **35.56 cm / 14 in**) into a **pixel-to-cm** scale;
3. estimates each bird's weight with the **slice-based volume model** from
   [BroilerWeightEstimation](https://github.com/ganhao89/BroilerWeightEstimation): the bird mask is rotated, cut into cylinders and summed, then multiplied by density;
4. **distills** the teacher into a lightweight **YOLOv8n-seg** student that can run on a **Raspberry Pi**;
5. validates both against scale readings exported by the **FlockLabel** GUI. FlockLabel is a separate tool; this package only reads its exports.

A dedicated `broilerseg.viz` module draws every step: masks, calibration geometry, slices, 3D cylinders, error analysis, density sweeps, training curves and teacher-vs-student agreement.

---

## Install

```bash
git clone <this repo> && cd broilerseg
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # core + teachers + ultralytics + notebook extras
```

### Choosing the teacher

| `TEACHER` (notebook setting) | Models | Access | Licence |
|---|---|---|---|
| `"grounded_sam2"` (default) | Grounding DINO + SAM 2.1 | public, downloads straight away | Apache-2.0 |
| `"sam3"` | Meta SAM 3 | gated: request access at <https://huggingface.co/facebook/sam3>, then `from huggingface_hub import login; login()` | SAM License (acknowledge SAM 3 in publications) |

```python
from broilerseg.segmentation import load_teacher
teacher = load_teacher("grounded_sam2", dino="base", sam2="large")   # or load_teacher("sam3")
```

| Extra      | Installs                                         | Needed for                         |
|------------|--------------------------------------------------|------------------------------------|
| *(core)*   | numpy, opencv, pandas, matplotlib, seaborn       | calibration, volume, metrics, viz  |
| `sam`      | torch, transformers, accelerate, huggingface_hub | Grounded SAM 2 and SAM 3 teachers  |
| `yolo`     | ultralytics                                      | training, YOLO inference, export   |
| `notebook` | jupyterlab, ipykernel, ipywidgets, tqdm          | running the notebooks              |
| `dev`      | pytest, ruff, flake8                             | tests and linting                  |

From inside a running notebook you can also run `%pip install -e "..[all]"`.

### Running the teachers on a MacBook Pro (Apple silicon)

Meta's reference SAM 3 code needs CUDA and Triton, so it doesn't run on a Mac. `broilerseg` uses the **Hugging Face `transformers` port** (`Sam3Model`) instead; Grounding DINO and SAM 2.1 also run through `transformers`. That port runs on **MPS** (the Apple GPU), with `PYTORCH_ENABLE_MPS_FALLBACK=1` set automatically, and falls back to the CPU. Expect seconds per frame rather than milliseconds. Frames are downscaled to `max_side=1280` before inference, and the masks are returned at full resolution.

---

## The two notebooks

| Notebook | What it does | Data it asks for |
|---|---|---|
| `notebooks/01_sam3_segmentation_and_weight.ipynb` | Teacher masks (Grounded SAM 2 / SAM 3) -> **fitted** mass equation (leave-one-event-out + nested CV) -> weights, most accurate density, videos | **FlockLabel export** (weight validation data) |
| `notebooks/02_sam3_to_yolov8_distillation.ipynb` | Searches a **nested folder tree** of videos, labels frames with the teacher, builds a YOLO-seg dataset (split by video or folder), trains, evaluates and **saves** YOLOv8n-seg with NCNN/ONNX exports | **Top folder of video sub-folders** |
| `notebooks/03_yolo_segmentation_and_weight.ipynb` | Loads the YOLO model saved by Notebook 2, segments with it, fits the mass equation on YOLO masks, compares with Notebook 1, renders videos and writes the Raspberry Pi files | **Saved model folder** + **FlockLabel export** |

Each notebook **prompts you to paste a folder path**. To skip the prompt, set the environment variable `BROILERSEG_WEIGHT_DIR` or `BROILERSEG_VIDEO_DIR`. Set `DRY_RUN = True` in the first cell to run either notebook on generated demo data, with no models and no GPU. Figures are saved as numbered PNGs under `~/broilerseg_outputs/` (change `OUTPUT_DIR` in the Settings cell).

---

## Two-stage segmentation (pen crop)

```
full 4K frame --stage 1: "animal pen" prompt--> pen box --crop, full resolution-->
pen crop --stage 2: broiler / scale / feeder--> masks (pasted back to full-frame coordinates)
```

```python
from broilerseg.segmentation import load_teacher
from broilerseg.roi import CroppedSegmenter, PenDetector

teacher = load_teacher("sam3")                       # or "grounded_sam2"
seg = CroppedSegmenter(teacher, detector=PenDetector(teacher))
result = seg.segment(frame_rgb)                      # masks in full-frame coordinates
```

## Package layout

```
src/broilerseg/
  config.py          classes, text prompts, feeder diameter, density defaults
  io/                video folder scanning + FlockLabel export loader
  segmentation/      Grounded SAM 2 / SAM 3 teachers, YOLO student, mask NMS / clean-up / bird-on-scale selection
  calibration.py     pixel-to-cm from the feeder (3 methods + repository HSV fallback)
  volume.py          body-axis alignment, slice profile, cylinder volume, mass
  experiments.py     WeightExperiment: segment -> calibrate -> select bird -> volume -> kg
  evaluation.py      MAE/RMSE/MAPE, event medians, density x slice sweep, closed-form density
  labeling.py        auto-label video frames with SAM 3 (resumable)
  yolo_dataset.py    SAM 3 masks -> YOLO-seg polygons, split by video
  distill.py         train student, teacher-student agreement, benchmark, Pi export
  viz/               ALL plots (overlays, calibration, volume, dataset, metrics, training)
  synthetic.py       demo videos + FlockLabel-style export + colour segmenter
  notebook.py        folder prompt, environment report
  cli.py             `broilerseg label | build | train | export`
```

## Quick API tour

```python
from broilerseg.segmentation import Sam3Segmenter, select_bird
from broilerseg.calibration import calibrate_from_mask
from broilerseg.volume import estimate_volume
from broilerseg import viz

seg = Sam3Segmenter()                         # cuda > mps > cpu
res = seg.segment(frame_rgb)                  # broiler / scale / feeder instances
cal = calibrate_from_mask(res.best("feeder").mask)
bird, info = select_bird(res)                 # the bird standing on the scale
est = estimate_volume(bird.mask, cal.pixel_to_cm, density_kg_m3=900, slice_count=None)
print(est.mass_kg)
viz.plot_volume_storyboard(frame_rgb, res, bird, est)
```

Command line, for long runs:

```bash
broilerseg label  /path/videos  labels/ --frames-per-video 20
broilerseg build  labels/  yolo_ds/
broilerseg train  yolo_ds/data.yaml --epochs 100
broilerseg export runs/broilerseg/student/weights/best.pt
```

## Documentation

* [`docs/methodology.pdf`](docs/methodology.pdf): illustrated methodology (pipeline, calibration, slice model, fitted mass equation, cross-validation)
* [`docs/method.md`](docs/method.md): segmentation, calibration and volume model, with equations
* [`docs/data_formats.md`](docs/data_formats.md): FlockLabel exports, label folders, YOLO dataset
* [`docs/raspberry_pi.md`](docs/raspberry_pi.md): deploying the student
* [`CONTRIBUTING.md`](CONTRIBUTING.md): code style (PEP 8, numpydoc), tests

## Testing

```bash
pip install -e ".[dev]"
pytest            # runs on synthetic data; no GPU or model downloads
ruff check src tests && flake8 --max-line-length 99 src tests
```

## License

MIT (same as the upstream BroilerWeightEstimation repository).
