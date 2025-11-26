# models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship

from database import Base


class Frame(Base):
    __tablename__ = "frames"

    id = Column(Integer, primary_key=True, index=True)
    video_path = Column(String, index=True)
    frame_path = Column(String, unique=True, index=True)
    timestamp = Column(Float, index=True)

    detections = relationship("Detection", back_populates="frame", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer, ForeignKey("frames.id", ondelete="CASCADE"))
    label = Column(String, index=True)
    score = Column(Float)
    bbox = Column(JSON)  # {"x":..., "y":..., "w":..., "h":...}

    frame = relationship("Frame", back_populates="detections")