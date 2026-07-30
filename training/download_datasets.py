from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASET_TARGETS = [
    ("saumyapatel/traffic-vehicles-object-detection", "data/raw/traffic_vehicles_object_detection"),
    ("nadinpethiyagoda/vehicle-dataset-for-yolo", "data/raw/vehicle_dataset_for_yolo"),
    ("pkdarabi/vehicle-detection-image-dataset", "data/raw/vehicle_detection_image_dataset"),
    ("vishnu0399/emergency-vehicle-siren-sounds", "data/raw/emergency_vehicle_siren_sounds"),
    ("abhiramasdf/emergency-vehicle-sirens-with-traffic-noise", "data/raw/emergency_vehicle_sirens_with_traffic_noise"),
    ("aalborguniversity/aau-rainsnow", "data/raw/aau_rainsnow"),
]


def main() -> None:
    kaggle_json = PROJECT_ROOT / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(f"Missing Kaggle credentials file: {kaggle_json}")

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(PROJECT_ROOT)

    for dataset, target in DATASET_TARGETS:
        target_path = PROJECT_ROOT / target
        target_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "kaggle.cli",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(target_path),
            "--unzip",
        ]
        print(f"[download] {dataset} -> {target_path}")
        subprocess.run(cmd, check=True, env=env)

    print("Dataset download complete.")


if __name__ == "__main__":
    main()
