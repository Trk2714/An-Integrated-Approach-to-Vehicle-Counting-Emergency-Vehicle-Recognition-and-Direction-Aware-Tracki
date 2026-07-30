from __future__ import annotations

import json
import mimetypes
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.database import EmergencyAlert, VideoInferenceRun, get_db, init_db
from inference.audio_inference import AudioSirenClassifier
from inference.media_utils import extract_audio_track
from inference.video_inference import TrafficVideoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "inference" / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

VEHICLE_MODEL_PATH = PROJECT_ROOT / "models" / "vehicle_detector_best.pt"
EMERGENCY_MODEL_PATH = PROJECT_ROOT / "models" / "emergency_visual_best.pt"
AUDIO_MODEL_PATH = PROJECT_ROOT / "models" / "audio_siren_resnet18.pt"

ALLOWED_INFERENCE_DEVICES = {"auto", "cpu", "cuda"}

video_processor: TrafficVideoProcessor | None = None
audio_classifier: AudioSirenClassifier | None = None


def normalize_inference_device(preference: str | None) -> str:
    value = (preference or "auto").strip().lower()
    if value not in ALLOWED_INFERENCE_DEVICES:
        return "auto"
    return value


REQUESTED_INFERENCE_DEVICE = normalize_inference_device(os.getenv("INFERENCE_DEVICE", "auto"))


app = FastAPI(
    title="Smart Traffic Management API",
    description="Vehicle counting, tracking, emergency detection, and congestion-aware signal prioritization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def save_upload(upload: UploadFile, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return target


def run_video_url(run_id: int) -> str:
    return f"/runs/{run_id}/video"


def serialize_run(run: VideoInferenceRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "created_at": run.created_at.isoformat(),
        "input_video": run.input_video,
        "output_video": run.output_video,
        "output_video_url": run_video_url(run.id),
        "fps": run.fps,
        "total_frames": run.total_frames,
        "duration_seconds": run.duration_seconds,
        "direction_counts": json.loads(run.direction_counts_json),
        "class_counts": json.loads(run.class_counts_json),
        "congestion": {
            "level": run.congestion_level,
            "average_vehicles_per_frame": run.congestion_average,
            "peak_vehicles_in_frame": run.congestion_peak,
        },
        "signal_priority": run.signal_priority,
    }


@app.on_event("startup")
def startup_event() -> None:
    global video_processor, audio_classifier
    init_db()

    missing = [p for p in (VEHICLE_MODEL_PATH, EMERGENCY_MODEL_PATH, AUDIO_MODEL_PATH) if not p.exists()]
    if missing:
        missing_text = ", ".join(str(p) for p in missing)
        raise RuntimeError(
            "Model files are missing. Train models first using training/run_all_training.py. "
            f"Missing: {missing_text}"
        )

    video_processor = TrafficVideoProcessor(
        vehicle_model_path=VEHICLE_MODEL_PATH,
        emergency_model_path=EMERGENCY_MODEL_PATH,
        audio_model_path=AUDIO_MODEL_PATH,
        inference_device=REQUESTED_INFERENCE_DEVICE,
    )
    audio_classifier = AudioSirenClassifier(AUDIO_MODEL_PATH, device=REQUESTED_INFERENCE_DEVICE)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": bool(video_processor is not None and audio_classifier is not None),
        "inference_device_requested": REQUESTED_INFERENCE_DEVICE,
        "video_inference_device": getattr(video_processor, "device_name", None),
        "audio_inference_device": getattr(audio_classifier, "device_name", None),
    }


@app.post("/infer/audio")
async def infer_audio(audio: UploadFile = File(...)) -> dict[str, Any]:
    if audio_classifier is None:
        raise HTTPException(status_code=503, detail="Audio model is not loaded.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RUNS_ROOT / timestamp
    audio_path = save_upload(audio, run_dir / audio.filename)

    try:
        result = audio_classifier.predict_file(audio_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Audio inference failed: {exc}") from exc

    return {
        "audio_path": str(audio_path),
        **result,
    }


@app.post("/infer/video")
async def infer_video(
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if video_processor is None:
        raise HTTPException(status_code=503, detail="Video processor is not loaded.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RUNS_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path = save_upload(video, run_dir / video.filename)
    audio_path = extract_audio_track(video_path, run_dir / f"{video_path.stem}_audio.wav")
    output_path = run_dir / f"processed_{video_path.stem}.mp4"

    try:
        summary = video_processor.process(video_path=video_path, output_path=output_path, audio_path=audio_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Video inference failed: {exc}") from exc

    run = VideoInferenceRun(
        input_video=summary["input_video"],
        output_video=summary["output_video"],
        fps=summary["fps"],
        total_frames=summary["total_frames"],
        duration_seconds=summary["duration_seconds"],
        direction_counts_json=json.dumps(summary["direction_counts"]),
        class_counts_json=json.dumps(summary["tracked_vehicle_count_by_class"]),
        congestion_level=summary["congestion"]["level"],
        congestion_average=summary["congestion"]["average_vehicles_per_frame"],
        congestion_peak=summary["congestion"]["peak_vehicles_in_frame"],
        signal_priority=summary["signal_priority"],
    )
    db.add(run)
    db.flush()

    alert = EmergencyAlert(
        run_id=run.id,
        visual_detected=summary["emergency"]["visual_detected"],
        siren_detected=summary["emergency"]["siren_detected"],
        siren_probability=summary["emergency"]["siren_probability"],
        confirmed=summary["emergency"]["confirmed"],
    )
    db.add(alert)
    db.commit()
    db.refresh(run)
    db.refresh(alert)

    return {
        "run_id": run.id,
        "summary": {**summary, "output_video_url": run_video_url(run.id)},
        "output_video_url": run_video_url(run.id),
        "alert": {
            "id": alert.id,
            "visual_detected": alert.visual_detected,
            "siren_detected": alert.siren_detected,
            "siren_probability": alert.siren_probability,
            "confirmed": alert.confirmed,
        },
    }


@app.get("/vehicle-counts")
def vehicle_counts(limit: int = 20, db: Session = Depends(get_db)) -> dict[str, Any]:
    runs = (
        db.query(VideoInferenceRun)
        .order_by(VideoInferenceRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return {"count": len(runs), "runs": [serialize_run(run) for run in runs]}


@app.get("/emergency-alerts")
def emergency_alerts(limit: int = 20, db: Session = Depends(get_db)) -> dict[str, Any]:
    alerts = (
        db.query(EmergencyAlert)
        .order_by(EmergencyAlert.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "count": len(alerts),
        "alerts": [
            {
                "id": alert.id,
                "created_at": alert.created_at.isoformat(),
                "run_id": alert.run_id,
                "visual_detected": alert.visual_detected,
                "siren_detected": alert.siren_detected,
                "siren_probability": alert.siren_probability,
                "confirmed": alert.confirmed,
            }
            for alert in alerts
        ],
    }


@app.get("/runs/{run_id}/video")
def get_run_video(run_id: int, db: Session = Depends(get_db)) -> FileResponse:
    run = db.query(VideoInferenceRun).filter(VideoInferenceRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    video_path = Path(run.output_video)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Processed video is missing for run {run_id}.")

    media_type = mimetypes.guess_type(video_path.name)[0] or "application/octet-stream"
    return FileResponse(path=video_path, media_type=media_type, filename=video_path.name)
