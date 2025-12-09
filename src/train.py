import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import yaml

from src.dataset import get_dataloaders
from src.models import BioCLIPClassifier
from src.peft_methods import apply_lora, apply_flylora

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
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--dataset_name", type=str, default="plankton", choices=["plankton", "insects2"])
    parser.add_argument("--method", type=str, default="linear_probe", choices=["linear_probe", "lora", "flylora"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    # Note: num_classes will be determined after loading dataset, but we need model for transform
    # So we load a temp model or just the transform first? 
    # BioCLIPClassifier loads the whole thing. Let's load it with a dummy num_classes first to get transform
    # Or better, instantiate it later.
    
    # We need transform for dataset
    # Let's assume standard CLIP transform for now or instantiate model to get it.
    print("Initializing model to get transforms...")
    temp_model = BioCLIPClassifier(num_classes=1) # Dummy
    preprocess = temp_model.get_preprocess()
    del temp_model
    
    # Load Data
    print(f"Loading {args.dataset_name} dataset from {args.data_dir}...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        seed=args.seed,
        transform=preprocess
    )
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    
    # Initialize Model
    model = BioCLIPClassifier(num_classes=num_classes)
    
    # Apply Method
    if args.method == "lora":
        print("Applying LoRA...")
        model = apply_lora(model)
    elif args.method == "flylora":
        print("Applying FlyLoRA...")
        model = apply_flylora(model)
    else:
        print("Using Linear Probe (default)...")
        # Ensure visual encoder is frozen (already done in BioCLIPClassifier init)
        # Only classifier head is trainable
        
    model = model.to(device)
    
    # Optimizer
    # Only optimize trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    best_val_acc = 0.0
    save_path = Path(args.output_dir) / args.dataset_name / args.method
    save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
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
