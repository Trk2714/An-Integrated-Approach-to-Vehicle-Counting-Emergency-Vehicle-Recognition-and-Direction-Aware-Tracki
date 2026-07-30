from __future__ import annotations

import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common import write_json


RANDOM_SEED = 42
VEHICLE_CLASS_NAMES = ["car", "threewheel", "bus", "truck", "motorbike", "van", "other"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Sample:
    source: str
    image_path: Path
    label_path: Path


def source_paths() -> list[tuple[str, Path, Path]]:
    traffic_root = PROJECT_ROOT / "data" / "raw" / "traffic_vehicles_object_detection" / "Traffic Dataset"
    yolo_root = PROJECT_ROOT / "data" / "raw" / "vehicle_dataset_for_yolo" / "vehicle dataset"

    return [
        ("traffic_train", traffic_root / "images" / "train", traffic_root / "labels" / "train"),
        ("traffic_val", traffic_root / "images" / "val", traffic_root / "labels" / "val"),
        ("yolo_train", yolo_root / "train" / "images", yolo_root / "train" / "labels"),
        ("yolo_val", yolo_root / "valid" / "images", yolo_root / "valid" / "labels"),
    ]


def collect_samples() -> list[Sample]:
    samples: list[Sample] = []
    for source_name, image_dir, label_dir in source_paths():
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing expected dataset folder: {image_dir} or {label_dir}")

        for image_path in image_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_path = image_path.relative_to(image_dir).with_suffix(".txt")
            label_path = label_dir / relative_path
            if not label_path.exists():
                continue
            samples.append(Sample(source=source_name, image_path=image_path, label_path=label_path))
    return samples


def remap_class_id(source: str, class_id: int) -> int | None:
    if source.startswith("yolo_"):
        if class_id > 5:
            return None
        return class_id
    if source.startswith("traffic_"):
        if class_id < 0 or class_id > 6:
            return None
        if class_id <= 5:
            return class_id
        return 6
    return None


def transform_label_file(source: str, label_path: Path) -> list[str]:
    lines_out: list[str] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_c, y_c, w, h = map(float, parts[1:])
        except ValueError:
            continue

        mapped = remap_class_id(source, class_id)
        if mapped is None:
            continue
        lines_out.append(f"{mapped} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    return lines_out


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
    train_samples, temp_samples = train_test_split(
        samples, test_size=0.3, random_state=RANDOM_SEED, shuffle=True
    )
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, random_state=RANDOM_SEED, shuffle=True
    )
    return train_samples, val_samples, test_samples


def unique_file_name(sample: Sample) -> str:
    digest = hashlib.sha1(str(sample.image_path).encode("utf-8")).hexdigest()[:12]
    return f"{sample.source}_{sample.image_path.stem}_{digest}{sample.image_path.suffix.lower()}"


def materialize_split(samples: list[Sample], split_name: str, output_root: Path) -> dict[str, int]:
    image_out_dir = output_root / split_name / "images"
    label_out_dir = output_root / split_name / "labels"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    non_empty_labels = 0
    for sample in samples:
        label_lines = transform_label_file(sample.source, sample.label_path)
        if not label_lines:
            continue

        file_name = unique_file_name(sample)
        image_target = image_out_dir / file_name
        label_target = label_out_dir / Path(file_name).with_suffix(".txt")
        shutil.copy2(sample.image_path, image_target)
        label_target.write_text("\n".join(label_lines), encoding="utf-8")
        written += 1
        non_empty_labels += 1

    return {"images": written, "labels": non_empty_labels}


def write_dataset_yaml(output_root: Path, yaml_path: Path) -> None:
    yaml_payload = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": VEHICLE_CLASS_NAMES,
    }
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    random.seed(RANDOM_SEED)
    output_root = PROJECT_ROOT / "data" / "processed" / "vehicle_yolo"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = collect_samples()
    if not samples:
        raise RuntimeError("No vehicle samples were discovered in raw datasets.")

    train_samples, val_samples, test_samples = split_samples(samples)
    train_stats = materialize_split(train_samples, "train", output_root)
    val_stats = materialize_split(val_samples, "val", output_root)
    test_stats = materialize_split(test_samples, "test", output_root)

    yaml_path = PROJECT_ROOT / "data" / "processed" / "vehicle_yolo.yaml"
    write_dataset_yaml(output_root, yaml_path)

    summary = {
        "num_raw_samples": len(samples),
        "splits": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "class_names": VEHICLE_CLASS_NAMES,
        "yaml_path": str(yaml_path),
    }
    write_json(PROJECT_ROOT / "data" / "processed" / "vehicle_yolo_stats.json", summary)
    print("Vehicle dataset prepared.")
    print(summary)


if __name__ == "__main__":
    main()
