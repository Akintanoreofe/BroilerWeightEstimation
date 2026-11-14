"""Generate synthetic NVR footage for trying FlockLabel.

Writes short clips named like the converted NVR exports
(``2025-10-14_07-40-00_to_07-40-06__ch00__SWF_1000000000001.mp4``) over three
days. In each clip a bird walks onto a scale, stands still, then walks off;
the scale display reads 0 until the bird is on it. An orange feeder of known
size (14 in) is in view, so BroilerWeightEstimation can calibrate from it.
``demo_ground_truth.json`` lists the true weights and on/off frames.

    python tools/make_demo_data.py demo_videos
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 1280, 720, 10, 6
SCALE = (440, 400, 380, 190)          # x, y, w, h of the scale platform
FEEDER = ((1110, 580), 70)            # centre, radius (140 px = 14 in)


def draw_bird(img: np.ndarray, cx: int, cy: int, size: float, phase: float) -> None:
    """Draw a top-down broiler: body ellipse, head, a slight wobble when walking."""
    body = (int(95 * size), int(62 * size))
    ang = 6 * np.sin(phase)
    cv2.ellipse(img, (cx, cy), body, ang, 0, 360, (232, 232, 228), -1, cv2.LINE_AA)
    hx = cx + int(body[0] * 0.95)
    cv2.circle(img, (hx, cy + int(4 * np.sin(phase))), int(26 * size), (236, 236, 232), -1, cv2.LINE_AA)
    cv2.circle(img, (hx + int(24 * size), cy), int(7 * size), (40, 40, 200), -1, cv2.LINE_AA)  # red, not feeder orange


def frame_at(i: int, n: int, on: int, off: int, weight_g: float, size: float, ts: datetime,
             rng: np.random.Generator) -> np.ndarray:
    img = np.full((H, W, 3), (48, 52, 56), np.uint8)
    noise = rng.integers(0, 10, (H, W, 1), dtype=np.uint8)
    img = cv2.add(img, np.repeat(noise, 3, axis=2))
    x, y, w, h = SCALE
    cv2.rectangle(img, (x, y), (x + w, y + h), (92, 96, 100), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 124, 128), 2)
    cv2.circle(img, FEEDER[0], FEEDER[1], (0, 128, 255), -1, cv2.LINE_AA)
    cv2.circle(img, FEEDER[0], FEEDER[1] - 22, (20, 70, 140), -1, cv2.LINE_AA)
    # bird path: walk in from the left, stand on the scale, walk off to the right
    sx, sy = x + w // 2 - 20, y + h // 2
    if i < on:
        t = i / max(on, 1)
        cx = int(-120 + t * (sx + 120))
    elif i <= off:
        cx = sx + int(3 * np.sin(i / 3))
    else:
        t = (i - off) / max(n - off, 1)
        cx = int(sx + t * (W + 200 - sx))
    phase = i * 0.9 if (i < on or i > off) else 0.0
    if -150 < cx < W + 150:
        draw_bird(img, cx, sy, size, phase)
    reading = weight_g if on + 2 <= i <= off - 1 else 0.0
    cv2.rectangle(img, (x + w + 20, y + 10), (x + w + 190, y + 60), (15, 25, 15), -1)
    cv2.putText(img, f"{reading / 1000:6.3f} kg", (x + w + 30, y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (60, 255, 90), 2, cv2.LINE_AA)
    cv2.putText(img, f"{ts:%m-%d-%Y %H:%M:%S}", (W - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (230, 230, 230), 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    truth = []
    n = FPS * SECONDS
    serial = 1000000000000
    for day in range(3):
        for hour, minute in ((7, 40), (9, 12), (15, 5), (16, 59)):
            start = datetime(2025, 10, 14 + day, hour, minute, 45)
            end = start + timedelta(seconds=SECONDS)
            serial += 1
            name = f"{start:%Y-%m-%d_%H-%M-%S}_to_{end:%H-%M-%S}__ch00__SWF_{serial}.mp4"
            weight = float(rng.integers(1800, 3200))
            size = 0.85 + (weight - 1800) / 1400 * 0.35
            on, off = int(n * 0.3), int(n * 0.7)
            vw = cv2.VideoWriter(str(out / name), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
            for i in range(n):
                vw.write(frame_at(i, n, on, off, weight, size, start + timedelta(seconds=i / FPS), rng))
            vw.release()
            truth.append({"file": name, "weight_g": weight, "bird_on_scale_frames": [on + 2, off - 1],
                          "frame_numbers_are": "0-based (the app shows them +1)"})
            print("wrote", name)
    # one file with no date in its name: the app falls back to the file's own date
    odd = out / "misc_export_0001.mp4"
    vw = cv2.VideoWriter(str(odd), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * 3):
        vw.write(frame_at(i, FPS * 3, 999, 999, 0, 1.0, datetime(2025, 10, 15, 12), rng))
    vw.release()
    (out / "demo_ground_truth.json").write_text(json.dumps(
        {"note": "Synthetic demo. Feeder diameter 140 px = 14 in, so pixel_to_cm = 0.254.",
         "fps": FPS, "clips": truth}, indent=2))
    print(f"Done: {len(truth) + 1} clips in {out}")


if __name__ == "__main__":
    main()
