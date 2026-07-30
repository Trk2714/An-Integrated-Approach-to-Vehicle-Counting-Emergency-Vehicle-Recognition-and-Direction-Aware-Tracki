from __future__ import annotations

import shutil
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common import write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def locate_source_root() -> Path:
    candidates = [
        PROJECT_ROOT
        / "data"
        / "raw"
        / "vehicle_detection_image_dataset"
        / "No_Apply_Grayscale"
        / "No_Apply_Grayscale"
        / "Vehicles_Detection.v8i.yolov8",
        PROJECT_ROOT
        / "data"
        / "raw"
        / "vehicle_detection_image_dataset"
        / "Apply_Grayscale"
        / "Apply_Grayscale"
        / "Vehicles_Detection.v9i.yolov8",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate emergency visual YOLO source folder.")


def convert_labels_to_single_class(label_path: Path) -> str:
    lines_out: list[str] = []
    if not label_path.exists():
        return ""
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        coords = " ".join(parts[1:])
        lines_out.append(f"0 {coords}")
    return "\n".join(lines_out)


def main() -> None:
    source_root = locate_source_root()
    output_root = PROJECT_ROOT / "data" / "processed" / "emergency_visual_yolo"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train", "valid": "val", "test": "test"}
    split_stats: dict[str, dict[str, int]] = {}

    for source_split, target_split in split_map.items():
        image_dir = source_root / source_split / "images"
        label_dir = source_root / source_split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing split folders under {source_root}: {source_split}")

        image_out_dir = output_root / target_split / "images"
        label_out_dir = output_root / target_split / "labels"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        lbl_count = 0
        for image_path in image_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
            transformed_label = convert_labels_to_single_class(label_path)

            target_image = image_out_dir / image_path.name
            target_label = label_out_dir / image_path.with_suffix(".txt").name
            shutil.copy2(image_path, target_image)
            target_label.write_text(transformed_label, encoding="utf-8")

            img_count += 1
            if transformed_label.strip():
                lbl_count += 1

        split_stats[target_split] = {"images": img_count, "labels_with_boxes": lbl_count}

    yaml_path = PROJECT_ROOT / "data" / "processed" / "emergency_visual_yolo.yaml"
    yaml_payload = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": ["emergency_vehicle"],
    }
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")

    summary = {
        "source_root": str(source_root),
        "splits": split_stats,
        "yaml_path": str(yaml_path),
        "class_names": ["emergency_vehicle"],
    }
    write_json(PROJECT_ROOT / "data" / "processed" / "emergency_visual_stats.json", summary)
    print("Emergency visual dataset prepared.")
    print(summary)


if __name__ == "__main__":
    main()
