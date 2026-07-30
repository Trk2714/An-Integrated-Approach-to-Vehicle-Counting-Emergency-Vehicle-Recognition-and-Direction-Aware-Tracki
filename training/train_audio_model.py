from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common import get_gpu_details, write_json


RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    if pretrained:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def wav_to_mel_tensor(
    wav_path: Path,
    sample_rate: int,
    n_mels: int,
    target_frames: int,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> torch.Tensor:
    signal, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :target_frames]

    mel_db = mel_db.astype(np.float32)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_tensor = torch.from_numpy(mel_db).unsqueeze(0).repeat(3, 1, 1)
    return mel_tensor


class AudioSirenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sample_rate: int, n_mels: int, target_frames: int):
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_frames = target_frames

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = Path(row["audio_path"])
        label = int(row["label_id"])
        feature = wav_to_mel_tensor(
            wav_path=audio_path,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            target_frames=self.target_frames,
        )
        return feature, torch.tensor(label, dtype=torch.long)


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for features, labels in dataloader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, labels)

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(len(dataloader.dataset), 1)
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = float(avg_loss)
    return avg_loss, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train siren vs no-siren classifier.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--target-frames", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)
    gpu_details = get_gpu_details()
    print(f"GPU details: {gpu_details}")

    data_dir = PROJECT_ROOT / "data" / "processed" / "audio_siren"
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"
    for csv_path in (train_csv, val_csv, test_csv):
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required file: {csv_path}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_dataset = AudioSirenDataset(train_df, args.sample_rate, args.n_mels, args.target_frames)
    val_dataset = AudioSirenDataset(val_df, args.sample_rate, args.n_mels, args.target_frames)
    test_dataset = AudioSirenDataset(test_df, args.sample_rate, args.n_mels, args.target_frames)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda:0")
    model = create_resnet18(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_state = None
    best_val_f1 = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, device, criterion, optimizer=None)

        epoch_log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_log)
        print(epoch_log)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training failed to produce a best checkpoint.")

    model.load_state_dict(best_state)
    _, test_metrics = run_epoch(model, test_loader, device, criterion, optimizer=None)

    model_path = PROJECT_ROOT / "models" / "audio_siren_resnet18.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "class_to_idx": {"no_siren": 0, "siren": 1},
            "sample_rate": args.sample_rate,
            "n_mels": args.n_mels,
            "target_frames": args.target_frames,
        },
        model_path,
    )

    metrics_payload = {
        "gpu": gpu_details,
        "train_args": vars(args),
        "best_model_path": str(model_path),
        "best_val_f1": float(best_val_f1),
        "test_metrics": test_metrics,
        "history": history,
    }
    write_json(PROJECT_ROOT / "models" / "audio_siren_metrics.json", metrics_payload)
    print("Audio model training complete.")
    print(metrics_payload)


if __name__ == "__main__":
    main()
