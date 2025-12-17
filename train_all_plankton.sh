#!/bin/bash

# Plankton Training Script
# Epochs: 3

echo "Starting Plankton Training..."

# 1. LoRA (Batch Size 100)
echo "Training LoRA..."
python src/train.py --dataset_name plankton --data_dir data/plankton --method lora --epochs 3 --batch_size 100

# 2. FlyLoRA (Batch Size 100)
echo "Training FlyLoRA..."
python src/train.py --dataset_name plankton --data_dir data/plankton --method flylora --epochs 3 --batch_size 100

# 3. DoRA (Batch Size 60)
echo "Training DoRA..."
python src/train.py --dataset_name plankton --data_dir data/plankton --method dora --epochs 3 --batch_size 60

# 4. PiSSA (Batch Size 80)
echo "Training PiSSA..."
python src/train.py --dataset_name plankton --data_dir data/plankton --method pissa --epochs 3 --batch_size 80

echo "Plankton Training Complete!"
