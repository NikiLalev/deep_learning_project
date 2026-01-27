import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from tqdm import tqdm


from src.model.classification_imagenet import YOLOPretrain

# --- MOCK IMPORTS FOR DATA LOADERS ---
# You should replace these with: from src.preprocessing.loaders import get_dataloaders
# For this script to run standalone in this example, I am using torchvision placeholders.
from torchvision import datasets, transforms

# ==========================================
# CONFIGURATION
# ==========================================

# Set this to True to use Optuna, False to use the MANUAL_CONFIG below
USE_OPTUNA = False 

# Fixed parameters for Manual Mode
MANUAL_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "optimizer": "Adam",  # Options: "Adam", "SGD"
    "epochs": 10,
    "num_classes": 20     # Pascal VOC has 20 classes
}

# General Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"   # Path to your dataset

# ==========================================
# DATA LOADING UTILS
# ==========================================

def get_data_loaders(batch_size):
    """
    Replace this function with your actual loader from src/preprocessing/loaders.py
    """
    # TODO: Connect this to your src.preprocessing.loaders
    # Example placeholder using CIFAR10 or FakeData just to make code runnable:
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # YOLO input size
        transforms.ToTensor(),
    ])
    
    # Placeholder: Replace with your Pascal VOC or ImageNet loader
    train_set = datasets.FakeData(size=1000, image_size=(3, 224, 224), num_classes=20, transform=transform)
    val_set = datasets.FakeData(size=200, image_size=(3, 224, 224), num_classes=20, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# ==========================================
# TRAINING LOOPS
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

# ==========================================
# OPTUNA OBJECTIVE
# ==========================================

def objective(trial):
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    
    # 2. Setup Data
    train_loader, val_loader = get_data_loaders(batch_size)
    
    # 3. Setup Model
    # Note: Ensure num_classes matches your dataset
    model = YOLOPretrain(num_classes=MANUAL_CONFIG["num_classes"]).to(DEVICE)
    
    # 4. Setup Optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    # We use a smaller number of epochs for optimization search usually
    epochs = 5 
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Report intermediate objective value to Optuna
        trial.report(val_acc, epoch)

        # Handle pruning (stop unpromising trials early)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc

# ==========================================
# MANUAL EXECUTION
# ==========================================

def run_manual_training():
    print(f"Starting Manual Training on {DEVICE}...")
    cfg = MANUAL_CONFIG
    
    train_loader, val_loader = get_data_loaders(cfg["batch_size"])
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
        
    # Save the model
    torch.save(model.state_dict(), "yolo_pretrain_manual.pth")
    print("Model saved to yolo_pretrain_manual.pth")

# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    if USE_OPTUNA:
        print("Starting Optuna Hyperparameter Search...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10) # Set n_trials to how many runs you want

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
    else:
        run_manual_training()