from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available. Aborting as GPU usage is mandatory.")
    return torch.device("cuda:0")


def get_gpu_details() -> dict[str, Any]:
    device = ensure_cuda()
    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    return {
        "device_index": idx,
        "name": props.name,
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "cuda_runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
