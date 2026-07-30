from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common import get_gpu_details, write_json


def extract_metrics(metrics_obj: object) -> dict[str, float]:
    results_dict = getattr(metrics_obj, "results_dict", {}) or {}
    out: dict[str, float] = {}
    for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        value = results_dict.get(key, 0.0)
        out[key] = float(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 vehicle detector on merged dataset.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_details = get_gpu_details()
    print(f"GPU details: {gpu_details}")

    data_yaml = PROJECT_ROOT / "data" / "processed" / "vehicle_yolo.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Vehicle dataset yaml not found: {data_yaml}")

    model = YOLO("yolov8n.pt")
    train_results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=4,
        project=str(PROJECT_ROOT / "models"),
        name="vehicle_detector",
        exist_ok=True,
        pretrained=True,
        patience=1000,
        verbose=True,
    )
    val_metrics = model.val(data=str(data_yaml), split="val", imgsz=args.imgsz, device=0)
    test_metrics = model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, device=0)

    save_dir = Path(train_results.save_dir)
    best_weight = save_dir / "weights" / "best.pt"
    last_weight = save_dir / "weights" / "last.pt"
    if not best_weight.exists():
        raise FileNotFoundError(f"Best weight was not generated at {best_weight}")

    out_best = PROJECT_ROOT / "models" / "vehicle_detector_best.pt"
    out_last = PROJECT_ROOT / "models" / "vehicle_detector_last.pt"
    shutil.copy2(best_weight, out_best)
    shutil.copy2(last_weight, out_last)

    metrics_payload = {
        "gpu": gpu_details,
        "train_args": vars(args),
        "save_dir": str(save_dir),
        "best_weight": str(out_best),
        "last_weight": str(out_last),
        "val_metrics": extract_metrics(val_metrics),
        "test_metrics": extract_metrics(test_metrics),
    }
    write_json(PROJECT_ROOT / "models" / "vehicle_detector_metrics.json", metrics_payload)
    print("Vehicle detector training complete.")
    print(metrics_payload)


if __name__ == "__main__":
    main()
