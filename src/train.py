import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add project root to sys.path to allow running script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import get_dataloaders
from src.models import BioCLIPClassifier
from src.peft_utils import apply_lora, apply_flylora, apply_linear_probe, apply_dora, apply_pissa

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description="BioCLIP 2 PEFT Training")
    parser.add_argument("--config", type=str, help="Path to config yaml file")
    parser.add_argument("--data_dir", type=str, help="Path to dataset")
    parser.add_argument("--dataset_name", type=str, choices=["plankton", "insects2"])
    parser.add_argument("--method", type=str, choices=["linear_probe", "lora", "flylora", "dora", "pissa"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--flylora_k", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--use_full_data", action="store_true", help="Use full dataset instead of few-shot")
    args = parser.parse_args()
       
    # Helper to get value from args
    def get_arg(name, default=None):
        val = getattr(args, name)
        if val is not None and val is not False:
             if isinstance(val, bool) and val is False:
                 return default
             return val
        return default

    # Set values
    data_dir = get_arg("data_dir")
    if data_dir is None:
        raise ValueError("data_dir must be provided via args or config")
        
    dataset_name = get_arg("dataset_name", "plankton")
    method = get_arg("method", "linear_probe")
    epochs = get_arg("epochs", 50)
    batch_size = get_arg("batch_size", 100)
    lr = get_arg("lr", 1e-3)
    lora_rank = get_arg("lora_rank", 16)
    flylora_k = get_arg("flylora_k", 8)
    dropout = get_arg("dropout", 0.1)
    weight_decay = get_arg("weight_decay", 0.01)
    use_full_data = args.use_full_data
        
    output_dir = get_arg("output_dir", "experiments")
    seed = get_arg("seed", 42)
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset_name} ({data_dir})")
    print(f"  Method: {method}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  LR: {lr}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Use Full Data: {use_full_data}")
    if method == "lora":
        print(f"  LoRA Rank: {lora_rank}")
        print(f"  Dropout: {dropout}")
    if method == "flylora":
        print(f"  LoRA Rank: {lora_rank}")
        print(f"  FlyLoRA k: {flylora_k}")
        print(f"  Dropout: {dropout}")
    if method == "dora":
        print(f"  DoRA Rank: {lora_rank}")
        print(f"  Dropout: {dropout}")
    if method == "pissa":
        print(f"  PiSSA Rank: {lora_rank}")
        print(f"  Dropout: {dropout}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    temp_model = BioCLIPClassifier(num_classes=1) # Dummy
    preprocess = temp_model.get_preprocess()
    del temp_model
    
    # Load Data
    print(f"Loading {dataset_name} dataset from {data_dir}...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir, 
        batch_size=batch_size, 
        seed=seed,
        transform=preprocess,
        use_full_data=use_full_data
    )
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    
    # Initialize Model
    model = BioCLIPClassifier(num_classes=num_classes)
    
    # Apply Method
    if method == "lora":
        print("Applying LoRA...")
        model = apply_lora(model, r=lora_rank, dropout=dropout)
    elif method == "flylora":
        print("Applying FlyLoRA...")
        model = apply_flylora(model, r=lora_rank, k=flylora_k, dropout=dropout)
    elif method == "dora":
        print("Applying DoRA...")
        model = apply_dora(model, r=lora_rank, dropout=dropout)
    elif method == "pissa":
        print("Applying PiSSA...")
        model = apply_pissa(model, r=lora_rank, dropout=dropout)
    else:
        print("Using Linear Probe (default)...")
        model = apply_linear_probe(model)
        
    model = model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    best_val_acc = 0.0
    save_path = Path(output_dir) / dataset_name / method
    save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path / "best_model.pth")
            print("Saved best model.")
            
    print("\nTraining Complete.")
    print("Evaluating on Test Set...")
    
    # Load best model
    model.load_state_dict(torch.load(save_path / "best_model.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
