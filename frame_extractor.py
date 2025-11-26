# frame_extractor.py
from pathlib import Path
from typing import List, Tuple

import cv2


def extract_frames(video_path: str, out_dir: str, step: int = 5) -> List[Tuple[str, float]]:
    """
    Extract every `step`-th frame from the video.
    Returns a list of (frame_path, timestamp) tuples.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results: List[Tuple[str, float]] = []

    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            timestamp = frame_idx / fps
            out_name = f"frame_{Path(video_path).stem}_{saved_idx:05d}.jpg"
            out_path = out_dir_path / out_name
            cv2.imwrite(str(out_path), frame)
            results.append((str(out_path), timestamp))
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return results