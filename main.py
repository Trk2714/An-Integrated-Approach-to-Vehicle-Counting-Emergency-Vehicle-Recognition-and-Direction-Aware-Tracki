from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def wait_until_keyboard_interrupt(processes: list[subprocess.Popen]) -> None:
    try:
        while True:
            time.sleep(1)
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(f"Subprocess exited unexpectedly with code {proc.returncode}.")
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


def run_services() -> None:
    python = sys.executable

    backend_cmd = [
        python,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    frontend_cmd = [
        python,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "frontend" / "app.py"),
        "--server.port",
        "8501",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    backend_proc = subprocess.Popen(backend_cmd, cwd=PROJECT_ROOT)
    time.sleep(3)
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=PROJECT_ROOT)

    print("Backend:  http://127.0.0.1:8000")
    print("Swagger:  http://127.0.0.1:8000/docs")
    print("Frontend: http://127.0.0.1:8501")
    wait_until_keyboard_interrupt([backend_proc, frontend_proc])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start local Smart Traffic backend + frontend.")
    parser.add_argument(
        "--setup-train",
        action="store_true",
        help="Run full preprocessing + training before starting services.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="When used with --setup-train, download Kaggle datasets first.",
    )
    parser.add_argument("--vehicle-epochs", type=int, default=5)
    parser.add_argument("--emergency-epochs", type=int, default=5)
    parser.add_argument("--audio-epochs", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    if args.setup_train:
        cmd = [
            python,
            str(PROJECT_ROOT / "training" / "run_all_training.py"),
            "--vehicle-epochs",
            str(args.vehicle_epochs),
            "--emergency-epochs",
            str(args.emergency_epochs),
            "--audio-epochs",
            str(args.audio_epochs),
        ]
        if args.download:
            cmd.append("--download")
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    run_services()


if __name__ == "__main__":
    main()
