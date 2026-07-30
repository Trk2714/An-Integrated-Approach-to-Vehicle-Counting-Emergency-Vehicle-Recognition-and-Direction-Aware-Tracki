from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common import write_json


RANDOM_SEED = 42


def collect_wavs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*.wav") if p.is_file()])


def build_metadata() -> pd.DataFrame:
    records: list[dict[str, str | int]] = []

    ds1_root = PROJECT_ROOT / "data" / "raw" / "emergency_vehicle_siren_sounds" / "sounds"
    ds2_root = PROJECT_ROOT / "data" / "raw" / "emergency_vehicle_sirens_with_traffic_noise"

    siren_dirs = [
        ds1_root / "ambulance",
        ds1_root / "firetruck",
        ds2_root / "Dataset" / "Dataset" / "ambulance",
        ds2_root / "Dataset" / "Dataset" / "firetruck",
        ds2_root / "Dataset" / "Dataset" / "police",
    ]
    no_siren_dirs = [
        ds1_root / "traffic",
        ds2_root / "traffic" / "traffic",
        ds2_root / "Dataset" / "Dataset" / "traffic",
    ]

    for folder in siren_dirs:
        for wav in collect_wavs(folder):
            records.append(
                {
                    "audio_path": str(wav),
                    "label": "siren",
                    "label_id": 1,
                    "source_folder": str(folder),
                }
            )
    for folder in no_siren_dirs:
        for wav in collect_wavs(folder):
            records.append(
                {
                    "audio_path": str(wav),
                    "label": "no_siren",
                    "label_id": 0,
                    "source_folder": str(folder),
                }
            )

    if not records:
        raise RuntimeError("No audio files discovered in expected Kaggle folders.")

    df = pd.DataFrame(records).drop_duplicates(subset=["audio_path"]).reset_index(drop=True)
    return df


def save_splits(df: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label_id"],
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label_id"],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "metadata.csv", index=False)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    return {
        "all": int(len(df)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "processed" / "audio_siren"
    if output_dir.exists():
        for file_path in output_dir.glob("*.csv"):
            file_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_metadata()
    split_sizes = save_splits(df, output_dir)
    label_counts = df["label"].value_counts().to_dict()

    summary = {
        "split_sizes": split_sizes,
        "label_counts": label_counts,
        "label_mapping": {"no_siren": 0, "siren": 1},
        "metadata_path": str(output_dir / "metadata.csv"),
    }
    write_json(output_dir / "stats.json", summary)
    print("Audio metadata prepared.")
    print(summary)


if __name__ == "__main__":
    main()
