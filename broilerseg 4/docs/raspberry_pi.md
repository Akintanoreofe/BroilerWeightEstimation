# Running the student on a Raspberry Pi

The segmentation model has to run on a Raspberry Pi next to the coop camera. SAM 3 (about 0.85 B parameters) cannot, but a distilled **YOLOv8n-seg** (about 3.4 M parameters) can.

## 1. Export (on the training machine)

```python
from broilerseg.distill import export_for_raspberry_pi
export_for_raspberry_pi("runs/broilerseg/student/weights/best.pt", imgsz=640)
# -> best_ncnn_model/  (recommended on ARM CPUs)  and  best.onnx
```

or `broilerseg export path/to/best.pt --imgsz 640`.

**Train and export at the same `imgsz` you will use on the Pi.** 320 or 480 px is several times faster than 640 px. Check the IoU and weight error at that size in Notebook 2 before deploying.

## 2. Install on the Pi (64-bit Raspberry Pi OS, Pi 4 or 5)

```bash
python -m venv ~/bw && source ~/bw/bin/activate
pip install ultralytics ncnn
pip install -e /path/to/broilerseg        # core only; no SAM 3 needed
```

## 3. Inference loop

```python
import cv2
from broilerseg.segmentation import YoloSegmenter, select_bird, clean_mask
from broilerseg.calibration import calibrate_from_mask
from broilerseg.volume import estimate_volume

DENSITY, SLICES = 900.0, None           # use the values chosen in Notebook 1
seg = YoloSegmenter("best_ncnn_model", imgsz=640)
cap = cv2.VideoCapture("rtsp://camera")   # or a file
px_to_cm = None
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    res = seg.segment(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    feeder = res.best("feeder")
    if px_to_cm is None and feeder is not None:          # camera is fixed: calibrate once
        px_to_cm = calibrate_from_mask(feeder.mask).pixel_to_cm
    bird, info = select_bird(res)
    if bird is not None and px_to_cm:
        kg = estimate_volume(clean_mask(bird.mask), px_to_cm, DENSITY, SLICES).mass_kg
        print(f"{kg:.2f} kg  (on-scale fraction {info['on_scale']:.2f})")
```

## Tips

* You don't need an estimate for every frame. Sample a few frames per second and report the **median** per weighing event, which is how the validation metrics are computed.
* Re-calibrate occasionally (e.g. hourly, median of several frames) in case the camera gets bumped.
* Time the model on the Pi itself. The speeds in Notebook 2 come from the training machine.
