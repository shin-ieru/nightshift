# process_video.py
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from database import SessionLocal
from models import Frame, Detection
from frame_extractor import extract_frames
from yolo_detector import run_yolo_on_image


def process_video(video_path: str, frame_step: int = 5):
    """
    Full pipeline:
      - extract frames into frames/
      - run YOLO
      - write detections + frames into SQLite
    """
    frames_dir = "frames"
    Path(frames_dir).mkdir(exist_ok=True)

    db: Session = SessionLocal()
    try:
        frame_infos = extract_frames(video_path, frames_dir, step=frame_step)

        for frame_path, ts in frame_infos:
            img = cv2.imread(frame_path)
            if img is None:
                continue

            frame_row = Frame(
                video_path=str(video_path),
                frame_path=Path(frame_path).name,
                timestamp=ts,
            )
            db.add(frame_row)
            db.flush()  # get frame_row.id

            detections = run_yolo_on_image(img, conf=0.3)
            for det in detections:
                db_det = Detection(
                    frame_id=frame_row.id,
                    label=det["label"],
                    score=det["score"],
                    bbox=det["bbox"],
                )
                db.add(db_det)

        db.commit()
    finally:
        db.close()