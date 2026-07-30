from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(step_name: str, command: list[str]) -> None:
    print(f"\n[step] {step_name}")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full local training pipeline (GPU mandatory).")
    parser.add_argument("--download", action="store_true", help="Download Kaggle datasets before preprocessing.")
    parser.add_argument("--vehicle-epochs", type=int, default=5)
    parser.add_argument("--emergency-epochs", type=int, default=5)
    parser.add_argument("--audio-epochs", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    run_step("GPU check", [python, "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"])
    run_step("Dependency check", [python, "-c", "import ultralytics, fastapi, streamlit, librosa, sklearn; print('Dependencies OK')"])

    if args.download:
        run_step("Kaggle dataset download", [python, "training/download_datasets.py"])

    run_step("Prepare vehicle dataset", [python, "training/prepare_vehicle_data.py"])
    run_step("Prepare emergency visual dataset", [python, "training/prepare_emergency_visual_data.py"])
    run_step("Prepare audio metadata", [python, "training/prepare_audio_data.py"])

    run_step(
        "Train vehicle detector",
        [python, "training/train_vehicle_detector.py", "--epochs", str(args.vehicle_epochs)],
    )
    run_step(
        "Train emergency visual detector",
        [python, "training/train_emergency_visual.py", "--epochs", str(args.emergency_epochs)],
    )
    run_step(
        "Train audio classifier",
        [python, "training/train_audio_model.py", "--epochs", str(args.audio_epochs)],
    )

    print("\nFull training pipeline finished successfully.")


if __name__ == "__main__":
    main()
