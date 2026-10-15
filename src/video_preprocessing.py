"""Interactive video preprocessing utility for the estimation pipeline.

Crops raw videos to a user-selected region and time range,
collects weight metadata, and saves results to test_videos/.
When run as a script, it invokes :func:`main`.

Usage:
    Single video:
        python video_preprocessing.py --input ../raw_videos/cam1.mp4

    Batch mode (same crop for all videos):
        python video_preprocessing.py --input_dir ../raw_videos/

    With time range:
        python video_preprocessing.py --input ../raw_videos/cam1.mp4 --start 00:00:05 --end 00:00:30
"""

import argparse
import csv
import os
import sys
import tkinter as tk
from pathlib import Path

import cv2

from VideoIOClass import VideoReader, VideoWriter


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "test_videos"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"


def parse_time(time_str: str) -> float:
    """Convert a colon-delimited time or seconds value to seconds.

    Parameters
    ----------
    time_str : str
        Time as ``HH:MM:SS``, ``MM:SS``, or a number of seconds. ``None``
        returns ``None``.

    Returns
    -------
    float or None
        Parsed time in seconds.

    Raises
    ------
    ValueError
        If the string contains a value that cannot be converted to a number.
    """
    if time_str is None:
        return None
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(time_str)


def format_time(seconds: float) -> str:
    """Format a number of seconds as hours, minutes, and fractional seconds.

    Parameters
    ----------
    seconds : float
        Time in seconds.

    Returns
    -------
    str
        Time formatted as ``HH:MM:SS.ss``.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def select_time_range_interactive(video_path: str) -> tuple:
    """Use a GUI to select a start and end time in a video.

    The preview is paused initially; the slider browses frames, and the
    controls set the start/end points or play the video before confirmation.

    Parameters
    ----------
    video_path : str
        Input video path.

    Returns
    -------
    tuple of float
        Selected ``(start_time, end_time)`` in seconds. The values are swapped
        if the chosen start is later than the chosen end.

    Raises
    ------
    RuntimeError
        If the video cannot be opened.
    """
    import threading
    from tkinter import ttk
    from PIL import Image, ImageTk

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # State
    state = {
        "start_time": 0.0,
        "end_time": duration,
        "playing": False,
        "current_frame": 0,
        "confirmed": False,
    }

    # --- Build the GUI ---
    root = tk.Tk()
    root.title("Time Selection")

    # Get screen size and scale preview to fit
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_preview_w = screen_w - 100
    max_preview_h = screen_h - 300  # leave room for controls
    scale = min(max_preview_w / orig_w, max_preview_h / orig_h, 1.0)
    preview_w = int(orig_w * scale)
    preview_h = int(orig_h * scale)

    # Video display label
    video_label = tk.Label(root)
    video_label.pack(padx=5, pady=5)

    # Info label
    info_var = tk.StringVar(value="Drag the slider to browse frames")
    info_label = tk.Label(root, textvariable=info_var, font=("Arial", 10))
    info_label.pack()

    # Slider
    slider_var = tk.IntVar(value=0)
    slider = ttk.Scale(root, from_=0, to=max(total_frames - 1, 1),
                       orient="horizontal", variable=slider_var)
    slider.pack(fill="x", padx=20, pady=5)

    # Time display
    time_frame = tk.Frame(root)
    time_frame.pack(pady=5)
    start_var = tk.StringVar(value=f"Start: {format_time(0.0)}")
    end_var = tk.StringVar(value=f"End: {format_time(duration)}")
    tk.Label(time_frame, textvariable=start_var, font=("Arial", 10), fg="green").pack(side="left", padx=20)
    tk.Label(time_frame, textvariable=end_var, font=("Arial", 10), fg="red").pack(side="left", padx=20)

    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    def set_start():
        """Set the selected start time to the currently displayed frame."""
        t = state["current_frame"] / fps if fps > 0 else 0
        state["start_time"] = t
        start_var.set(f"Start: {format_time(t)}")
        print(f"Start time set: {format_time(t)}")

    def set_end():
        """Set the selected end time to the currently displayed frame."""
        t = state["current_frame"] / fps if fps > 0 else 0
        state["end_time"] = t
        end_var.set(f"End: {format_time(t)}")
        print(f"End time set: {format_time(t)}")

    def toggle_play():
        """Toggle preview playback and update the button label."""
        state["playing"] = not state["playing"]
        play_btn.config(text="Pause" if state["playing"] else "Play")

    def confirm():
        """Stop the preview, mark the selection confirmed, and close the GUI."""
        state["playing"] = False
        state["confirmed"] = True
        if state.get("after_id"):
            root.after_cancel(state["after_id"])
        root.destroy()

    tk.Button(btn_frame, text="Set Start", command=set_start, width=10,
              bg="#4CAF50", fg="white").pack(side="left", padx=5)
    play_btn = tk.Button(btn_frame, text="Play", command=toggle_play, width=10)
    play_btn.pack(side="left", padx=5)
    tk.Button(btn_frame, text="Set End", command=set_end, width=10,
              bg="#f44336", fg="white").pack(side="left", padx=5)
    tk.Button(btn_frame, text="Confirm", command=confirm, width=10,
              bg="#2196F3", fg="white").pack(side="left", padx=5)

    # Read and display a frame
    def show_frame_at(frame_idx):
        """Display a frame and refresh the current-frame and time labels.

        Parameters
        ----------
        frame_idx : int
            Index of the frame to display.
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return
        state["current_frame"] = frame_idx
        # Convert BGR -> RGB, resize, display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (preview_w, preview_h))
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # prevent garbage collection
        video_label.config(image=imgtk)
        # Update info
        t = frame_idx / fps if fps > 0 else 0
        info_var.set(f"Frame: {frame_idx}/{total_frames - 1}  |  Time: {format_time(t)}")

    # Show first frame
    show_frame_at(0)

    # Track slider changes (when user drags it while paused)
    last_slider_val = [0]

    def update_loop():
        """Advance playback or respond to slider movement until confirmation."""
        if state["confirmed"]:
            return
        if state["playing"]:
            next_frame = state["current_frame"] + 1
            if next_frame >= total_frames:
                next_frame = 0
            state["current_frame"] = next_frame
            slider_var.set(next_frame)
            show_frame_at(next_frame)
            delay = max(int(1000 / fps), 1) if fps > 0 else 33
            state["after_id"] = root.after(delay, update_loop)
        else:
            # Check if slider was dragged
            val = slider_var.get()
            if val != last_slider_val[0]:
                last_slider_val[0] = val
                show_frame_at(val)
            state["after_id"] = root.after(50, update_loop)

    state["after_id"] = root.after(50, update_loop)

    print(f"Video duration: {format_time(duration)} ({total_frames} frames @ {fps:.1f} fps)")
    print("Use the GUI to select start/end times.")
    root.mainloop()

    cap.release()

    start_time = state["start_time"]
    end_time = state["end_time"]

    if start_time > end_time:
        start_time, end_time = end_time, start_time
        print(f"Swapped start/end: {format_time(start_time)} -> {format_time(end_time)}")

    print(f"Selected range: {format_time(start_time)} - {format_time(end_time)}")
    return start_time, end_time


def get_screen_size() -> tuple:
    """Get the display resolution through tkinter.

    Returns
    -------
    tuple of int
        Screen width and height in pixels.
    """
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    return screen_w, screen_h


def select_crop_region(video_path: str, start_time: float) -> tuple:
    """Display a video frame and let the user select a crop rectangle.

    The selected rectangle is mapped back to original frame coordinates. If
    the selection is canceled, the full frame is returned.

    Parameters
    ----------
    video_path : str
        Input video path.
    start_time : float
        Time in seconds at which to display the frame for selection.

    Returns
    -------
    tuple of int
        Crop rectangle ``(x, y, width, height)`` in original frame pixels.

    Raises
    ------
    RuntimeError
        If the video cannot be opened or the requested frame cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame at {format_time(start_time)}")

    orig_h, orig_w = frame.shape[:2]

    # Scale frame to fit screen with some margin
    screen_w, screen_h = get_screen_size()
    margin = 100  # leave some margin for window decorations
    max_w = screen_w - margin
    max_h = screen_h - margin

    scale = min(max_w / orig_w, max_h / orig_h, 1.0)  # never upscale
    display_w = int(orig_w * scale)
    display_h = int(orig_h * scale)

    if scale < 1.0:
        display_frame = cv2.resize(frame, (display_w, display_h))
        print(f"Frame resized from {orig_w}x{orig_h} to {display_w}x{display_h} (scale={scale:.2f})")
    else:
        display_frame = frame
        print(f"Frame fits screen at original size {orig_w}x{orig_h}")

    print("Draw a bounding box on the frame. Press Enter/Space to confirm, 'c' to cancel.")
    roi = cv2.selectROI("Select Crop Region", display_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("No crop region selected. Using full frame.")
        return 0, 0, orig_w, orig_h

    # Map ROI back to original frame coordinates
    if scale < 1.0:
        x = int(roi[0] / scale)
        y = int(roi[1] / scale)
        w = int(roi[2] / scale)
        h = int(roi[3] / scale)
        # Clamp to frame bounds
        x = min(x, orig_w - 1)
        y = min(y, orig_h - 1)
        w = min(w, orig_w - x)
        h = min(h, orig_h - y)
    else:
        x, y, w, h = roi

    print(f"Crop region (original coords): x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h


def ask_weight() -> float:
    """Ask for a broiler weight using a tkinter dialog.

    Returns
    -------
    float
        Entered weight in kilograms, or ``0.0`` if the dialog closes without
        a valid value.
    """
    weight = [None]

    root = tk.Tk()
    root.title("Broiler Weight Input")
    root.geometry("300x150")
    root.resizable(False, False)

    tk.Label(root, text="Enter broiler chicken weight (kg):", font=("Arial", 11)).pack(pady=(15, 5))

    entry = tk.Entry(root, font=("Arial", 12), justify="center")
    entry.pack(pady=5)
    entry.focus_set()

    def on_ok(event=None):
        """Parse the entered weight or display an invalid-number message.

        Parameters
        ----------
        event : tkinter event or None, optional
            Event supplied when the Return key triggers the callback.
        """
        try:
            val = float(entry.get())
            weight[0] = val
            root.destroy()
        except ValueError:
            entry.delete(0, tk.END)
            entry.insert(0, "Invalid number")

    tk.Button(root, text="OK", command=on_ok, width=10).pack(pady=10)
    root.bind("<Return>", on_ok)

    root.mainloop()

    if weight[0] is None:
        print("No weight entered. Defaulting to 0.0 kg.")
        return 0.0

    print(f"Weight entered: {weight[0]} kg")
    return weight[0]


def save_metadata(video_name: str, weight_kg: float):
    """Append a video name and weight to the metadata CSV file.

    Parameters
    ----------
    video_name : str
        Name of the processed video.
    weight_kg : float
        Recorded weight in kilograms.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = METADATA_CSV.exists()

    with open(METADATA_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video_name", "weight_kg"])
        writer.writerow([video_name, weight_kg])

    print(f"Metadata saved: {video_name}, {weight_kg} kg")


def process_video(video_path: str, roi: tuple, start_time: float, end_time: float, weight_kg: float):
    """Crop a video's frames to a region and time interval, then save metadata.

    Parameters
    ----------
    video_path : str
        Source video path.
    roi : tuple of int
        Crop rectangle ``(x, y, width, height)`` in source-frame coordinates.
    start_time : float
        Start time in seconds.
    end_time : float
        End time in seconds.
    weight_kg : float
        Weight to record for the resulting video.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    output_path = str(OUTPUT_DIR / f"{video_name}.mp4")

    reader = VideoReader(video_path)
    fps, orig_w, orig_h = reader.get_info()

    x, y, w, h = roi
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total = end_frame - start_frame

    writer = VideoWriter(output_path, fps, (w, h))

    reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = 0

    print(f"Processing: {video_name}")
    print(f"  Frames {start_frame} -> {end_frame} ({total} frames)")
    print(f"  Crop region: ({x}, {y}, {w}, {h})")

    while True:
        current = int(reader.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current >= end_frame:
            break

        ret, frame = reader.read()
        if not ret:
            break

        cropped = frame[y:y + h, x:x + w]
        writer.write(cropped)
        frame_count += 1

        if frame_count % 100 == 0 or frame_count == total:
            pct = frame_count / total * 100 if total > 0 else 100
            print(f"  Progress: {frame_count}/{total} ({pct:.1f}%)")

    reader.release()
    writer.release()

    print(f"  Saved: {output_path} ({frame_count} frames)")

    save_metadata(f"{video_name}.mp4", weight_kg)


def main():
    """Parse command-line options and run interactive or batch preprocessing.

    Accepts either one input video or a directory of videos. A shared time
    range and crop are selected, then each video is cropped and its weight
    metadata is recorded.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess raw videos: crop spatially and temporally, record weight metadata."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to a single input video")
    group.add_argument("--input_dir", type=str, help="Directory of input videos for batch mode")
    parser.add_argument("--start", type=str, default=None, help="Start time (HH:MM:SS or seconds)")
    parser.add_argument("--end", type=str, default=None, help="End time (HH:MM:SS or seconds)")

    args = parser.parse_args()

    # Collect video paths
    if args.input:
        video_paths = [args.input]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Error: {args.input_dir} is not a directory")
            sys.exit(1)
        video_paths = sorted(
            str(p) for p in input_dir.iterdir()
            if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
        )
        if not video_paths:
            print(f"No video files found in {args.input_dir}")
            sys.exit(1)

    print(f"Videos to process: {len(video_paths)}")
    for vp in video_paths:
        print(f"  {vp}")

    # Step 1: Time range selection
    first_video = video_paths[0]
    if args.start is not None and args.end is not None:
        start_time = parse_time(args.start)
        end_time = parse_time(args.end)
        print(f"Time range from CLI: {format_time(start_time)} - {format_time(end_time)}")
    else:
        start_time, end_time = select_time_range_interactive(first_video)

    # Step 2: Crop region selection (once, applied to all)
    roi = select_crop_region(first_video, start_time)

    # Step 3 & 4: Process each video
    for vp in video_paths:
        weight = ask_weight()
        process_video(vp, roi, start_time, end_time, weight)

    print("\nDone! All videos processed.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata CSV: {METADATA_CSV}")


if __name__ == "__main__":
    main()
