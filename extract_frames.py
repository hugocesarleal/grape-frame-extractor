"""
extract_frames.py
-----------------
Extracts one representative frame per time window from vineyard videos,
prioritizing sharpness (Laplacian variance) and adequate brightness.

Usage:
    python extract_frames.py

Videos must be placed in the project root named video1.mp4 ... video7.mp4.
Extracted frames are saved to dataset_grapes_structured/<video_name>/.
"""

import cv2
import os
import numpy as np


# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

OUTPUT_FOLDER = "dataset_grapes_structured"
VIDEOS = [f"video{i}.mp4" for i in range(1, 8)]
WINDOWS_PER_SECOND = 4      # time windows per second — one frame extracted per window
BRIGHTNESS_THRESHOLD = 40.0  # frames below this mean brightness are considered too dark


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_sharpness(gray_frame: np.ndarray) -> float:
    """
    Returns the Laplacian variance of a grayscale frame.
    Higher values indicate sharper images with more defined edges.
    """
    return cv2.Laplacian(gray_frame, cv2.CV_64F).var()


def compute_brightness(gray_frame: np.ndarray) -> float:
    """
    Returns the mean pixel intensity of a grayscale frame (range 0–255).
    """
    return float(np.mean(gray_frame))


def select_best_frame(window: list, threshold: float) -> tuple:
    """
    Selects the most suitable frame from a time window.

    Primary criterion: among frames with brightness >= threshold,
    picks the sharpest one.

    Fallback: if all frames are too dark, picks the sharpest one
    regardless of brightness.

    Parameters
    ----------
    window : list
        List of frame dicts with keys 'image', 'sharpness',
        'brightness', and 'original_index'.
    threshold : float
        Minimum mean brightness to consider a frame well-lit.

    Returns
    -------
    tuple
        (best_frame_dict, status) where status is 'ideal' or 'fallback'.
    """
    bright_frames = [f for f in window if f["brightness"] >= threshold]

    if bright_frames:
        best = max(bright_frames, key=lambda f: f["sharpness"])
        return best, "ideal"

    best = max(window, key=lambda f: f["sharpness"])
    return best, "fallback"


def save_frame(frame: np.ndarray, path: str) -> None:
    """
    Saves a frame as a JPEG file at the given path.
    """
    cv2.imwrite(path, frame)


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, output_folder: str,
                   windows_per_sec: int = 4,
                   brightness_threshold: float = 40.0) -> int:
    """
    Processes a single video and extracts exactly one frame per time window.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_folder : str
        Folder where extracted frames will be saved.
    windows_per_sec : int
        Number of time windows per second of video.
    brightness_threshold : float
        Minimum mean brightness (0–255) for a frame to be considered well-lit.

    Returns
    -------
    int
        Number of frames saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open: {video_path}")
        return 0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    window_size = max(1, int(round(original_fps / windows_per_sec)))

    print(f"  Video FPS: {original_fps:.2f} | Window size: {window_size} frames")

    window = []
    frame_index = 0
    frames_saved = 0
    fallbacks = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        window.append({
            "image": frame,
            "sharpness": compute_sharpness(gray),
            "brightness": compute_brightness(gray),
            "original_index": frame_index,
        })

        if len(window) == window_size:
            best, status = select_best_frame(window, brightness_threshold)

            filename = (
                f"frame_{frames_saved:06d}"
                f"_orig_{best['original_index']:06d}.jpg"
            )
            save_frame(best["image"], os.path.join(output_folder, filename))

            if status == "fallback":
                fallbacks += 1

            frames_saved += 1
            window = []

        frame_index += 1

    # Handle remaining frames if the video does not end on a full window
    if window:
        best, status = select_best_frame(window, brightness_threshold)
        filename = (
            f"frame_{frames_saved:06d}"
            f"_orig_{best['original_index']:06d}.jpg"
        )
        save_frame(best["image"], os.path.join(output_folder, filename))
        if status == "fallback":
            fallbacks += 1
        frames_saved += 1

    cap.release()

    print(f"  Frames extracted: {frames_saved} | Fallbacks (dark windows): {fallbacks}")
    return frames_saved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    total = 0

    for video_name in VIDEOS:
        output_path = os.path.join(
            OUTPUT_FOLDER,
            os.path.splitext(video_name)[0]
        )
        print(f"\n=== Processing {video_name} ===")

        if not os.path.isfile(video_name):
            print(f"  [WARNING] File not found, skipping.")
            continue

        total += extract_frames(
            video_path=video_name,
            output_folder=output_path,
            windows_per_sec=WINDOWS_PER_SECOND,
            brightness_threshold=BRIGHTNESS_THRESHOLD,
        )

    print(f"\n{'='*50}")
    print(f"Extraction complete. Total frames generated: {total}")
    print(f"Output folder: {os.path.abspath(OUTPUT_FOLDER)}")


if __name__ == "__main__":
    main()