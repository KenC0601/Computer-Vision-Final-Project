import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import MetaAlbumDataset, create_unseen_5shot_splits, create_few_shot_splits
from src.models import BioCLIPClassifier
from src.peft_utils import apply_lora, apply_flylora, apply_linear_probe
from src.evaluation_utils import train_linear_probe, evaluate, get_classifier
from torch.utils.data import Subset, DataLoader
import numpy as np

def run_5shot_eval(data_dir, method, model_path=None, lora_rank=8, flylora_k=4, seed=42, batch_size=128):
    """
    Runs the 5-shot evaluation pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Running 5-Shot Evaluation: {method} ===")
    
    # 1. Initialize Model
    print("Initializing BioCLIP backbone...")
    model = BioCLIPClassifier(num_classes=1) 
    preprocess = model.get_preprocess()
    
    # 2. Load Data
    full_dataset = MetaAlbumDataset(data_dir, transform=preprocess)
    num_classes = len(full_dataset.classes)
    
    # Update the classifier head for the correct number of classes
    print(f"Updating classifier head for {num_classes} classes...")
    model.classifier = nn.Linear(model.embed_dim, num_classes)
    model = model.to(device)
    
    # Determine Split Strategy
    if method == "linear_probe" and model_path is None:
        print("Using Random 5-Shot Splits (Baseline Protocol)...")
        # num_val=0 means we get train (5 shots) and test (rest)
        train_idx, _, test_idx = create_few_shot_splits(full_dataset, num_shots=5, num_val=0, seed=seed)
    else:
        train_idx, test_idx = create_unseen_5shot_splits(full_dataset, seed=seed)
    
    train_set = Subset(full_dataset, train_idx)
    test_set = Subset(full_dataset, test_idx)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 3. Apply Method Structure
    if method == "lora":
        model = apply_lora(model, r=lora_rank)
    elif method == "flylora":
        model = apply_flylora(model, r=lora_rank, k=flylora_k)
    elif method == "dora":
        from src.peft_utils import apply_dora
        model = apply_dora(model, r=lora_rank)
    elif method == "pissa":
        from src.peft_utils import apply_pissa
        model = apply_pissa(model, r=lora_rank)
    else:
        model = apply_linear_probe(model)
        
    model = model.to(device)
    
    # 4. Load Backbone Weights
    if model_path and os.path.exists(model_path):
        print(f"Loading backbone from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        # Robust loading
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Direct load failed, attempting filtered load: {e}")
            model_dict = model.state_dict()
            # Filter out classifier weights and mismatched keys
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'classifier' not in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    else:
        print("Using Pre-trained BioCLIP Backbone (Baseline)")

    # 5. Train Linear Probe (with timing)
    print("Training 5-shot classifier...")
    
    # Optimizer for Head Only
    try:
        classifier = get_classifier(model)
        # Ensure it's trainable
        for p in classifier.parameters():
            p.requires_grad = True
        head_params = [p for p in classifier.parameters() if p.requires_grad]
    except:
        # Fallback
        head_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]

    if not head_params:
         print("Warning: No trainable head parameters found. Unfreezing classifier...")
         classifier = get_classifier(model)
         for p in classifier.parameters():
             p.requires_grad = True
         head_params = [p for p in classifier.parameters() if p.requires_grad]

    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    train_linear_probe(model, train_loader, criterion, optimizer, device, epochs=50)
    end_time = time.time()
    training_time = end_time - start_time
    
    # 6. Evaluate
    print("Evaluating...")
    acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {acc:.2f}%")
    
    # 7. Save 5-Shot Model
    if model_path:
        save_dir = os.path.dirname(model_path)
        save_name = f"5shot_classifier_{method}.pth"
        save_path = os.path.join(save_dir, save_name)
        torch.save(model.state_dict(), save_path)
    else:
        # For baseline or if no model path provided
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        save_dir = os.path.join("experiments", dataset_name, "baseline")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "5shot_classifier.pth")
        torch.save(model.state_dict(), save_path)

    return acc, training_time

def main():
    parser = argparse.ArgumentParser(description="5-Shot Evaluation of Tuned Backbones")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint (optional, for baseline use None)")
    parser.add_argument("--method", type=str, choices=["linear_probe", "lora", "flylora", "dora", "pissa"], required=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--flylora_k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    acc, _ = run_5shot_eval(
        data_dir=args.data_dir,
        method=args.method,
        model_path=args.model_path,
        lora_rank=args.lora_rank,
        flylora_k=args.flylora_k,
        seed=args.seed
    )
    
    # 7. Save Results
    results_file = "results_5shot.csv"
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    
    # Create header if file doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("Dataset,Method,Accuracy\n")
            
    with open(results_file, "a") as f:
        f.write(f"{dataset_name},{args.method},{acc:.2f}\n")
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
