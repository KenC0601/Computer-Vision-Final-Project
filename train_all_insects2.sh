#!/bin/bash

# Insects 2 Training Script
# Epochs: 20

echo "Starting Insects 2 Training..."

# 1. LoRA (Batch Size 100)
echo "Training LoRA..."
python src/train.py --dataset_name insects2 --data_dir data/insects2 --method lora --epochs 20 --batch_size 100 --use_full_data

# 2. FlyLoRA (Batch Size 100)
echo "Training FlyLoRA..."
python src/train.py --dataset_name insects2 --data_dir data/insects2 --method flylora --epochs 20 --batch_size 100 --use_full_data

# 3. DoRA (Batch Size 60)
echo "Training DoRA..."
python src/train.py --dataset_name insects2 --data_dir data/insects2 --method dora --epochs 20 --batch_size 60 --use_full_data

# 4. PiSSA (Batch Size 80)
echo "Training PiSSA..."
python src/train.py --dataset_name insects2 --data_dir data/insects2 --method pissa --epochs 20 --batch_size 80 --use_full_data

echo "Insects 2 Training Complete!"
