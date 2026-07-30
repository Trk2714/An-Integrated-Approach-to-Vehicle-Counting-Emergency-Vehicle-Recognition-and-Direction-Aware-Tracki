from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Generator

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "backend" / "traffic.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class VideoInferenceRun(Base):
    __tablename__ = "video_inference_runs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    input_video = Column(String, nullable=False)
    output_video = Column(String, nullable=False)
    fps = Column(Float, nullable=False)
    total_frames = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    direction_counts_json = Column(Text, nullable=False)
    class_counts_json = Column(Text, nullable=False)
    congestion_level = Column(String, nullable=False)
    congestion_average = Column(Float, nullable=False)
    congestion_peak = Column(Integer, nullable=False)
    signal_priority = Column(String, nullable=False)

    alert = relationship("EmergencyAlert", back_populates="run", uselist=False, cascade="all, delete-orphan")


class EmergencyAlert(Base):
    __tablename__ = "emergency_alerts"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    run_id = Column(Integer, ForeignKey("video_inference_runs.id"), nullable=False, unique=True)
    visual_detected = Column(Boolean, nullable=False, default=False)
    siren_detected = Column(Boolean, nullable=False, default=False)
    siren_probability = Column(Float, nullable=False, default=0.0)
    confirmed = Column(Boolean, nullable=False, default=False)

    run = relationship("VideoInferenceRun", back_populates="alert")


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
