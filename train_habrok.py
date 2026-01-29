#!/usr/bin/env python
from __future__ import annotations
# -*- coding: utf-8 -*-
"""
YOLO posttraining on Habrok HPC
Replicates the original YOLO (2016) paper training procedure
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import torchvision.transforms.functional as F
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project root
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

# Import project modules
from src.model.classification_imagenet import YOLOPretrain
from src.model.detector import YOLOv1
from yolo_loss import YOLOv1Loss  # (kept import minimal; iou helpers not used here)


def print_env() -> None:
    print(f"Project Root: {CURRENT_DIR}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {props.total_memory / 1e9:.2f} GB")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Training configuration."""
    NUM_CLASSES: int = 20

    # Training parameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 90
    OPTIMIZER: str = "ADAM"

    # Data parameters
    TRAIN_N: Optional[int] = None
    VAL_N: Optional[int] = None
    NUM_WORKERS: int = 8

    # LR schedule
    USE_LR_SCHEDULER: bool = True
    LR_MILESTONES: Tuple[int, ...] = (30, 60)
    LR_GAMMA: float = 0.1

    # Checkpointing / logging
    SAVE_FREQUENCY: int = 20
    WEIGHT_DECAY: float = 1e-4
    USE_AMP: bool = True

    # Paths
    CHECKPOINT_DIR: Path = CURRENT_DIR / "checkpoints"
    LOG_DIR: Path = CURRENT_DIR / "logs"

    # Runtime
    # DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp: str = ""

    def __post_init__(self) -> None:
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_config(self) -> Path:
        d = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in self.__dict__.items()
        }
        out = self.LOG_DIR / f"config_{self.timestamp}.json"
        with open(out, "w") as f:
            # json.dump(d, f, indent=2)
            json.dump(d, f, indent=2, default=str)
        return out


# ============================================================================
# Data loading
# ============================================================================

def _collate_yolo(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    return {"images": images, "boxes": boxes, "labels": labels}


def _ensure_pil(img: Any) -> Image.Image:
    if isinstance(img, dict):
        b = img.get("bytes", None)
        if b is None:
            raise ValueError(f"Image dict has no bytes. keys={list(img.keys())}")
        img = Image.open(BytesIO(b))
    return img


def _to_torch(example: Dict[str, Any]) -> Dict[str, Any]:
    img = example["image"]

    # Batched case
    if isinstance(img, list):
        imgs = [_ensure_pil(im) for im in img]
        return {
            "image": torch.stack([F.to_tensor(im) for im in imgs]),
            "boxes": [torch.tensor(b, dtype=torch.float32) for b in example["boxes"]],
            "labels": [torch.tensor(l, dtype=torch.long) for l in example["labels"]],
        }

    # Single example
    img = _ensure_pil(img)
    return {
        "image": F.to_tensor(img),
        "boxes": torch.tensor(example["boxes"], dtype=torch.float32),
        "labels": torch.tensor(example["labels"], dtype=torch.long),
    }


def get_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print("\n" + "=" * 60)
    print("LOADING PASCAL VOC DATASET")
    print("=" * 60)

    # ds = load_from_disk("data/pascal_voc_yolo_448")
    ds = load_from_disk("/scratch/s4015843/data/pascal_voc_yolo_448")
    train_ds = ds["train"]
    val_ds = ds["test"]  # original code uses test as val

    # 80/20 split (deterministic with seed)
    split = train_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    val_ds   = split["test"]   # HF uses "test" key for the held-out split

    # Optional streaming support (note: train_test_split requires map-style dataset)
    if getattr(config, "STREAMING", False):
        train_ds = train_ds.to_iterable_dataset()
        val_ds   = val_ds.to_iterable_dataset()
        test_ds  = test_ds.to_iterable_dataset()

    train_ds = train_ds.with_transform(_to_torch)
    val_ds = val_ds.with_transform(_to_torch)
    test_ds = ds["test"].with_transform(_to_torch)

    pin = torch.cuda.is_available()
    is_streaming = getattr(config, "STREAMING", False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
        shuffle=not is_streaming,
        collate_fn=_collate_yolo,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
        shuffle=False,
        collate_fn=_collate_yolo,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
        shuffle=False,
        collate_fn=_collate_yolo,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Targets
# ============================================================================

def build_targets_yolov1(
    boxes_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    S: int = 7,
    B: int = 2,
    C: int = 20,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    boxes_list: list length batch, each is Tensor [Ni,4] in normalized xywh (0..1)
    labels_list: list length batch, each is Tensor [Ni]
    Returns: targets Tensor [B,S,S, C + B*5]
    Layout per cell:
      [class_onehot(C), conf1, x1, y1, w1, h1, conf2, x2, y2, w2, h2]
    """
    bs = len(boxes_list)
    targets = torch.zeros((bs, S, S, C + B * 5), device=device)

    for b in range(bs):
        boxes = boxes_list[b]
        labels = labels_list[b]
        if boxes.numel() == 0:
            continue

        for (xc, yc, w, h), cls in zip(boxes, labels):
            i = int(yc * S)
            j = int(xc * S)
            i = max(0, min(S - 1, i))
            j = max(0, min(S - 1, j))

            # One object per cell (paper limitation)
            if targets[b, i, j, C] == 1:
                continue

            targets[b, i, j, int(cls)] = 1.0

            x_cell = xc * S - j
            y_cell = yc * S - i

            # box1
            targets[b, i, j, C + 0] = 1.0
            targets[b, i, j, C + 1 : C + 5] = torch.tensor([x_cell, y_cell, w, h], device=device)

            # box2
            targets[b, i, j, C + 5] = 1.0
            targets[b, i, j, C + 6 : C + 10] = torch.tensor([x_cell, y_cell, w, h], device=device)

    return targets


# ============================================================================
# Training / validation
# ============================================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: Config,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    total_images = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]")
    for batch in pbar:
        images = batch["images"].to(config.DEVICE)
        boxes = [b.to(config.DEVICE) for b in batch["boxes"]]
        labels = [l.to(config.DEVICE) for l in batch["labels"]]

        targets = build_targets_yolov1(boxes, labels, S=7, B=2, C=20, device=config.DEVICE)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(images)  # [B,7,7,30]
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        total_images += bs
        pbar.set_postfix(loss=f"{running_loss / total_images:.4f}")

    return running_loss / total_images


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    config: Config,
    epoch: int,
) -> float:
    model.eval()
    running_loss = 0.0
    total_images = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Val]")
    for batch in pbar:
        images = batch["images"].to(config.DEVICE)
        boxes = [b.to(config.DEVICE) for b in batch["boxes"]]
        labels = [l.to(config.DEVICE) for l in batch["labels"]]

        targets = build_targets_yolov1(boxes, labels, S=7, B=2, C=20, device=config.DEVICE)
        preds = model(images)
        loss = criterion(preds, targets)

        bs = images.size(0)
        running_loss += loss.item() * bs
        total_images += bs
        pbar.set_postfix(loss=f"{running_loss / total_images:.4f}")

    return running_loss / total_images


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: Config,
) -> Path:
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }

    latest_path = config.CHECKPOINT_DIR / "checkpoint_latest.pth"
    torch.save(checkpoint, latest_path)

    if epoch % config.SAVE_FREQUENCY == 0:
        epoch_path = config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, epoch_path)
        print(f"✓ Checkpoint saved: {epoch_path}")

    return latest_path


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: Config,
    start_epoch: int = 1,
) -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")

    print("\n" + "=" * 60)
    print("STARTING TRAINING - YOLOv1 PASCAL VOC (DETECTION)")
    print("=" * 60)
    print(f"Total epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Optimizer: {config.OPTIMIZER}")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config, epoch)
        val_loss = validate(model, val_loader, criterion, config, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["lr"].append(float(current_lr))

        print(f"\nEpoch {epoch}/{config.EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, config)

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_path = config.CHECKPOINT_DIR / "checkpoint_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": float(val_loss),
                },
                best_path,
            )
            print(f"  ★ New best model! Val Loss: {val_loss:.4f} (saved to {best_path})")

        print("=" * 60 + "\n")

    history_path = config.LOG_DIR / f"training_history_{config.timestamp}.json"
    with open(history_path, "w") as f:
        # json.dump(history, f, indent=2)
        json.dump(history, f, indent=2, default=str)

    print("\n✓ Training completed!")
    print(f"✓ Best validation loss: {best_val_loss:.4f}")
    print(f"✓ Training history saved to {history_path}")

    return history


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print_env()
    config = Config()
    config.save_config()

    # ... [Data loader and Logging code remains the same] ...
    train_loader, val_loader, _ = get_data_loaders(config)

    print("\n" + "=" * 60)
    print("INITIALIZING MODEL & OPTIMIZER")
    print("=" * 60)

    # 1. Always initialize the base architecture first
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(config.LR_MILESTONES),
        gamma=config.LR_GAMMA,
    )

    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and torch.cuda.is_available() else None
    criterion = YOLOv1Loss(S=7, B=2, C=20).to(config.DEVICE)

    # 2. RESUME LOGIC
    start_epoch = 1
    latest_path = config.CHECKPOINT_DIR / "checkpoint_latest.pth"
    
    if latest_path.exists():
        print(f"\n>>> Found checkpoint: {latest_path}. Resuming...")
        checkpoint = torch.load(latest_path, map_location=config.DEVICE)
        
        model.load_state_dict(checkpoint["model_state_dict"], )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # 3. FIX: Move optimizer state tensors to GPU manually
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(config.DEVICE)

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        start_epoch = checkpoint["epoch"] + 1
        print(f">>> Successfully loaded. Resuming from Epoch {start_epoch}\n")
    else:
        print("\n>>> No checkpoint found. Loading ImageNet pretrain weights...")
        pretrain = YOLOPretrain(num_classes=1000)
        # Assuming this file is in your project root
        # pretrain.load_state_dict(torch.load("yolo_pretrain_manual2.pth", map_location=config.DEVICE))
        checkpoint_data = torch.load("yolo_pretrain_manual3.pth", map_location=config.DEVICE)
        pretrain.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
        model.load_pretrain_weights(pretrain)

    # 3. Move model to GPU after weights are loaded
    model = model.to(config.DEVICE)

    # 4. Start Training (CRITICAL: passing start_epoch)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        start_epoch=start_epoch,
    )

    # 5. Save final model
    final_model_path = CURRENT_DIR / "yolov1_voc.pth"
    torch.save(model.state_dict(), final_model_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Loss: {min(history['val_loss']):.4f}")
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Model saved to: {final_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
