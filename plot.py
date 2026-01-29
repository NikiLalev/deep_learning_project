#!/usr/bin/env python3
"""
Plot YOLO training history from a JSON file.

Expected JSON keys:
- "train_loss": list[float]
- "val_loss":   list[float]
Optional:
- "lr":         list[float]  (ignored here)

Outputs:
- plots/loss_train_vs_val.png
- plots/train_loss.png
- plots/val_loss.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path.resolve()}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "train_loss" not in data or "val_loss" not in data:
        raise KeyError('JSON must contain keys "train_loss" and "val_loss".')

    if not isinstance(data["train_loss"], list) or not isinstance(data["val_loss"], list):
        raise TypeError('"train_loss" and "val_loss" must be lists.')

    if len(data["train_loss"]) == 0 or len(data["val_loss"]) == 0:
        raise ValueError('"train_loss" and "val_loss" must be non-empty lists.')

    return data


def ensure_plots_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_plot(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_train_vs_val(train_loss, val_loss, out_dir: Path) -> None:
    epochs_train = list(range(1, len(train_loss) + 1))
    epochs_val = list(range(1, len(val_loss) + 1))

    fig = plt.figure()
    plt.plot(epochs_train, train_loss, label="Train loss")
    plt.plot(epochs_val, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot(fig, out_dir / "loss_train_vs_val.pdf")


def plot_single(series, title: str, ylabel: str, filename: str, out_dir: Path) -> None:
    epochs = list(range(1, len(series) + 1))

    fig = plt.figure()
    plt.plot(epochs, series)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    save_plot(fig, out_dir / filename)


def main() -> None:
    json_path = Path("training_history_20260129_110725.json")
    out_dir = Path("plots")

    history = load_history(json_path)
    ensure_plots_dir(out_dir)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    plot_train_vs_val(train_loss, val_loss, out_dir)
    plot_single(train_loss, "Training Loss", "Loss", "train_loss.pdf", out_dir)
    plot_single(val_loss, "Validation Loss", "Loss", "val_loss.pdf", out_dir)

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
