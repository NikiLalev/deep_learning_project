#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO ImageNet Pretraining on Habrok HPC
Replicates the original YOLO (2016) paper training procedure
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# Setup paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print(f"Project Root: {current_dir}")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Import project modules
from src.model.classification_imagenet import YOLOPretrain
from src.preprocessing.data_loader import load_imagenet_iterable

# Check CUDA
print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# Configuration - Exact YOLO Paper Settings
# ============================================================================

class Config:
    """Training configuration matching YOLO 2016 paper"""
    
    # Model parameters
    NUM_CLASSES = 1000  # ImageNet-1k
    
    # HuggingFace token for streaming ImageNet
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    # Training parameters (FROM YOLO PAPER)
    BATCH_SIZE = 128  # Standard for ImageNet (reduce to 64 if OOM)
    LEARNING_RATE = 0.1  # Paper uses 0.1 for SGD
    EPOCHS = 160  # Paper: 160 epochs for pretraining
    OPTIMIZER = "SGD"
    
    # SGD settings (FROM YOLO PAPER)
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # Data parameters
    TRAIN_N = None  # None = full ImageNet (1.28M images)
    VAL_N = None    # None = full validation (50K images)
    NUM_WORKERS = 4
    
    # Learning rate schedule
    # Paper doesn't specify exact schedule for pretraining
    # Standard ImageNet: decay at epochs 30, 60, 90, 120
    USE_LR_SCHEDULER = True
    LR_MILESTONES = [30, 60, 90, 120]  # Decay at these epochs
    LR_GAMMA = 0.1  # Multiply LR by 0.1 at each milestone
    
    # Checkpointing
    CHECKPOINT_DIR = current_dir / "checkpoints"
    SAVE_FREQUENCY = 5  # Save every 5 epochs
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mixed precision (optional, not in original paper but speeds up training)
    USE_AMP = True
    
    # Logging
    LOG_DIR = current_dir / "logs"
    
    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_config(self):
        """Save configuration to JSON"""
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items() 
                      if not k.startswith('_') and k != 'HF_TOKEN'}
        config_file = self.LOG_DIR / f"config_{self.timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✓ Config saved to {config_file}")

# ============================================================================
# Data Loading
# ============================================================================

def get_data_loaders(config):
    """Load ImageNet"""
    print("\n" + "="*60)
    print("LOADING IMAGENET DATASET")
    print("="*60)
    
    # Enable streaming!
    ds_dict = load_imagenet_iterable(
        streaming=True,   # <--- SET TO TRUE
        train_n=config.TRAIN_N,
        val_n=config.VAL_N,
        hf_token=config.HF_TOKEN
    )
    
    # For streaming, we cannot use shuffle=True in DataLoader
    # (The dataset is already shuffled via buffer)
    train_loader = DataLoader(
        ds_dict["train"],
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        ds_dict["validation"],
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader
# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, config, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        inputs = batch['image'].to(config.DEVICE)
        targets = batch['label'].to(config.DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config.USE_AMP and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, config, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Val]")
    
    with torch.no_grad():
        for batch in pbar:
            inputs = batch['image'].to(config.DEVICE)
            targets = batch['label'].to(config.DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, val_acc, config):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    
    # Save latest
    latest_path = config.CHECKPOINT_DIR / "checkpoint_latest.pth"
    torch.save(checkpoint, latest_path)
    
    # Save periodic checkpoint
    if epoch % config.SAVE_FREQUENCY == 0:
        epoch_path = config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, epoch_path)
        print(f"✓ Checkpoint saved: {epoch_path}")
    
    return latest_path

# ============================================================================
# Main Training Loop
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, config):
    """Main training loop"""
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("STARTING TRAINING - YOLO IMAGENET PRETRAINING")
    print("="*60)
    print(f"Total epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Optimizer: {config.OPTIMIZER}")
    print("="*60 + "\n")
    
    for epoch in range(1, config.EPOCHS + 1):
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config, epoch
        )
        
        # Validation
        val_loss, val_acc = validate(
            model, val_loader, criterion, config, epoch
        )
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, val_acc, config)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = config.CHECKPOINT_DIR / "checkpoint_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, best_path)
            print(f"  ★ New best model! Val Acc: {val_acc:.2f}% (saved to {best_path})")
        
        print("="*60 + "\n")
    
    # Save training history
    history_path = config.LOG_DIR / f"training_history_{config.timestamp}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training completed!")
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Training history saved to {history_path}")
    
    return history

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function"""
    
    # Initialize config
    config = Config()
    config.save_config()
    
    print("\n" + "="*60)
    print("YOLO IMAGENET PRETRAINING - HABROK HPC")
    print("Replicating YOLO (2016) Paper Settings")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Optimizer: {config.OPTIMIZER} (momentum={config.MOMENTUM}, weight_decay={config.WEIGHT_DECAY})")
    print(f"LR Schedule: MultiStepLR (milestones={config.LR_MILESTONES}, gamma={config.LR_GAMMA})")
    print("="*60)
    
    # Load data
    train_loader, val_loader = get_data_loaders(config)
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    model = YOLOPretrain(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: YOLOPretrain (First 20 conv layers from YOLO)")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("="*60)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (YOLO paper settings)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.LR_MILESTONES,
        gamma=config.LR_GAMMA
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and torch.cuda.is_available() else None
    
    # Train
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler, scaler,
        config
    )
    
    # Save final model
    final_model_path = current_dir / "yolo_pretrain_imagenet.pth"
    torch.save(model.state_dict(), final_model_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Model saved to: {final_model_path}")
    print("="*60)


if __name__ == "__main__":
    main()
