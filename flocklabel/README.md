# FlockLabel

A desktop app (macOS and Windows) for labelling NVR videos of broilers on a
scale. Pick a day, play a video, drag a box around the bird on the scale,
type the reading on the scale display. Then export the labelled frames as an ML
dataset, or as an evaluation set for `BroilerWeightEstimation`.

Each labelled frame holds **one box** (the bird on the scale) and **the scale
reading on that frame**. There is no tare step and no channel.

## Install (once)

Python 3.10 to 3.13. In Terminal, from the unzipped `flocklabel` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Start

```bash
cd ~/Downloads/flocklabel            # wherever you unzipped it
source .venv/bin/activate
python run_flocklabel.py
```

Quit with ⌘Q (Ctrl+Q on Windows), or Ctrl+C in the Terminal window.

## Label your videos

1. **Make a project.** File → New / open project → choose or create an
   empty folder, e.g. `Documents/flock_project`. Labels save there as you go.
2. **Import videos.** File → Import videos → Browse to the folder with the
   videos → Import. The preview shows the date and time read from each name,
   e.g. `2025-10-01_16-59-45_to_16-59-47__ch00__SWF_1000381605376.mp4` gives
   1 Oct 2025, 16:59:45 to 16:59:47.
3. **Open a video.** Click a bold day on the calendar, then a video in the
   list (or a block on the strip under the video). It loads in the
   background; 4K files are shown at screen size but boxes are saved in the
   video's full resolution.
4. **Find a frame with the bird on the scale.** Space plays and pauses,
   ← → step one frame, Shift+← → step 10. Playback follows the real frame
   times, so irregular NVR frame gaps play as recorded.
5. **Label it.** Drag a box around the bird. The reading field opens; type
   what the scale shows and press Enter. Pick g or kg next to the field.
6. **More frames of the same weighing.** Press → then **D**: the box and
   reading are copied from the last labelled frame. Drag the box if the bird
   moved, or press Enter and type a new reading if it changed.
7. **Next video.** Click **Done** (or **Skip** if there is no usable
   weighing), then Page Down.
8. **Export.** File → Export dataset, or Export evaluation set.

An orange box means the frame has a box but no reading yet; it is not
exported until it has one.

## Keyboard

| Key | Action |
| --- | --- |
| Space | Play / pause |
| ← / → | Previous / next frame |
| Shift + ← / → | Back / forward 10 frames |
| Home / End | First / last frame |
| , / . | Previous / next labelled frame |
| D | Copy box and reading from the last labelled frame |
| Enter | Type the reading (Enter again saves, Esc returns to the video) |
| Delete / Backspace | Remove this frame's label |
| Page Up / Page Down | Previous / next video |
| ⌘Z / ⌘⇧Z | Undo / redo |
| F | Fit the video to the window (mouse wheel zooms) |
| ⌘E | Export dataset |
| F1 | Getting started |

## Exports

**Dataset** (`File → Export dataset`): `images/` and YOLO `labels/` split into
train/val/test **by video**, `annotations_coco.json` with `weight_g` per box,
`manifest.csv` (one row per frame: box, reading, timestamp, the video's
median reading), and `sequences/<video>.json` with every labelled frame of a
video. Filters: only videos marked Done, a date range, a cap on frames per
video.

**Evaluation set** (`File → Export evaluation set`): one trimmed clip per
video in `test_videos/` with `metadata.csv` (median reading in kg), the
layout `BroilerWeightEstimation` reads, plus `ground_truth/gt_boxes.csv`
(each frame's own box and reading). Optionally crop all clips with
**Edit → Draw evaluation crop**, which must include the scale *and* the
orange feeder because the pipeline calibrates from it. You can also set
**Edit → Set pixel_to_cm** yourself.

Scoring (copy `eval_scripts/*.py` into the repo's `src/` folder):

```bash
cd BroilerWeightEstimation/src
python run_eval.py /path/to/eval_export
python evaluate_against_gt.py /path/to/eval_export/ground_truth/gt_boxes.csv \
       --results-dir /path/to/eval_export/results --out /path/to/eval_export/eval_out
python weight_validation.py /path/to/eval_export/eval_out/matched.csv
```

You get the detection rate, frame-level and per-video MAE/RMSE/bias/MAPE,
and the density that fits best (fit it on some videos, test on the rest).

## Try it without your footage

```bash
python tools/make_demo_data.py demo_videos
```

That makes 13 short synthetic clips named like yours. Import `demo_videos`
into a new project. `demo_videos/demo_ground_truth.json` has the true weights.

## Notes

- **Videos that only some players open:** many converted NVR files are H.265
  with an `hev1` tag, which QuickTime refuses. FlockLabel decodes them with
  the FFmpeg inside OpenCV, so they play here even when QuickTime will not.
- A project is one folder with `flocklabel.sqlite`. Copy it to move between
  computers. If the video folder moves, import it again from the new place.
- Projects made with earlier test versions of this app are not compatible;
  start a new project folder.
- Tests: `python -m pytest -q` (set `FLOCKLABEL_REAL_VIDEO=/path/to/a/video.mp4`
  to include one of your own files).
