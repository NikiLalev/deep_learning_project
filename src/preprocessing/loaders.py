import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from tqdm import tqdm

# --- 1. SETUP PATHS so Python finds 'src' ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

# --- 2. IMPORTS ---
from src.model.classification_imagenet import YOLOPretrain
# We use the ImageNet loader because YOLOPretrain is a Classifier, not a Detector
from src.preprocessing.loaders import load_imagenet_iterable 

# ==========================================
# CONFIGURATION
# ==========================================

USE_OPTUNA = False  # Set to True to automate hyperparameter search

# Fixed parameters for Manual Mode
MANUAL_CONFIG = {
    "batch_size": 16,     # Keep small if running locally
    "learning_rate": 1e-3,
    "optimizer": "Adam",
    "epochs": 5,
    "num_classes": 1000,   # ImageNet has 1000 classes
    
    # Data limits (from your loaders.py defaults)
    # Increase these numbers when you want to do the REAL full training
    "train_n": 5000,       
    "val_n": 500
}

# Auto-detect device (Mac MPS, CUDA, or CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Using Device: MPS (Mac Metal)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using Device: CUDA")
else:
    DEVICE = torch.device("cpu")
    print(f"Using Device: CPU")

# ==========================================
# DATA LOADING
# ==========================================

def get_data_loaders(batch_size, train_n, val_n):
    """
    Connects to src/preprocessing/loaders.py
    """
    print("Loading ImageNet Stream...")
    
    # We override the N parameters to control dataset size from here
    # Your loaders.py needs to be able to accept these args or you edit loaders.py
    # Based on your code, load_imagenet takes these args.
    from src.preprocessing.loaders import load_imagenet
    
    # We use load_imagenet directly so we can pass n arguments
    # Note: We need the transforms. Your load_imagenet calls preprocess_imagenet inside map.
    # But we need 'to_torch' eventually. 
    # Let's use load_imagenet_iterable but we might need to patch the counts if hardcoded.
    
    ds_dict = load_imagenet_iterable() 
    
    # HuggingFace IterableDatasets don't behave exactly like Pytorch Datasets
    # We wrap them in a DataLoader.
    
    # Note: Since they are streams, shuffle=True is handled inside the stream buffer, 
    # not here.
    train_loader = torch.utils.data.DataLoader(ds_dict["train"], batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(ds_dict["validation"], batch_size=batch_size)
    
    return train_loader, val_loader

# ==========================================
# TRAINING LOOPS
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        # HuggingFace yields a dict: {'image': tensor, 'label': tensor}
        inputs = batch['image'].to(DEVICE)
        targets = batch['label'].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update pbar description
        pbar.set_postfix({'loss': running_loss/total if total > 0 else 0})
        
    # Avoid division by zero
    if total == 0: return 0, 0
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(DEVICE)
            targets = batch['label'].to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    if total == 0: return 0, 0
    return running_loss / len(loader), 100. * correct / total

# ==========================================
# OPTUNA OBJECTIVE
# ==========================================

def objective(trial):
    # 1. Hyperparameters to search
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    
    # 2. Setup
    train_loader, val_loader = get_data_loaders(batch_size, MANUAL_CONFIG["train_n"], MANUAL_CONFIG["val_n"])
    model = YOLOPretrain(num_classes=MANUAL_CONFIG["num_classes"]).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    # 3. Train
    for epoch in range(3): # Fewer epochs for search
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_manual_training():
    print("Starting Manual Training...")
    cfg = MANUAL_CONFIG
    
    train_loader, val_loader = get_data_loaders(cfg["batch_size"], cfg["train_n"], cfg["val_n"])
    model = YOLOPretrain(num_classes=cfg["num_classes"]).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    if cfg["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=0.9)
        
    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
    torch.save(model.state_dict(), "yolo_pretrain.pth")
    print("Model saved to yolo_pretrain.pth")

if __name__ == "__main__":
    if USE_OPTUNA:
        print("Starting Optuna Search...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)
        print("Best params:", study.best_params)
    else:
        run_manual_training()