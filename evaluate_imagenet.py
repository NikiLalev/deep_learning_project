import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sys
import os

CHECKPOINT_PATH = "checkpoint_best-2.pth"
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5

def calculate_accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top K indices
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        # Check which predictions match the target
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Sum up correct predictions in the top-k row
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate():
    print(f"Running evaluation on {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    # 1. Load Data
    ds_dict = load_imagenet_iterable(streaming=True) 
    val_loader = DataLoader(
        ds_dict["validation"],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 2. Load Model
    model = YOLOPretrain(num_classes=1000)
    
    # Load weights
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully")
    else:
        print(f"Checkpoint file not found at {CHECKPOINT_PATH}")
        return

    model.to(DEVICE)
    model.eval()

    # 3. Evaluation Loop
    top1_avg = 0.0
    top5_avg = 0.0
    total_batches = 0
    
    print("Starting inference...")
    
    # Use tqdm
    pbar = tqdm(val_loader, desc="Evaluating")
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            images = batch['image'].to(DEVICE)
            targets = batch['label'].to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Calculate Accuracy
            acc1, acc5 = calculate_accuracy(outputs, targets, topk=(1, 5))
            
            # Update running averages
            top1_avg += acc1.item()
            top5_avg += acc5.item()
            total_batches += 1

            if i % 10 == 0:
                pbar.set_postfix({
                    "Top-1": f"{top1_avg / total_batches:.2f}%", 
                    "Top-5": f"{top5_avg / total_batches:.2f}%"
                })

    # Final Results
    final_top1 = top1_avg / total_batches
    final_top5 = top5_avg / total_batches

    print(f"\n{'='*40}")
    print(f"FINAL RESULTS")
    print(f"{'='*40}")
    print(f"Top-1 Accuracy: {final_top1:.2f}%")
    print(f"Top-5 Accuracy: {final_top5:.2f}%")
    print(f"{'='*40}")

if __name__ == "__main__":
    evaluate()