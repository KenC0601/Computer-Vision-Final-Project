#!/bin/bash

# Activate environment if needed
# source activate bioclip_peft
# OR use the venv directly
PYTHON_CMD=".venv/bin/python"
export PYTHONPATH=.

# Datasets
DATASETS=("plankton" "insects2")

# Methods
METHODS=("linear_probe" "lora" "flylora")

# Loop
for DATASET in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Running experiments for $DATASET"
    echo "========================================"
    
    DATA_DIR="data/$DATASET"
    
    for METHOD in "${METHODS[@]}"; do
        echo "----------------------------------------"
        echo "Training $METHOD on $DATASET"
        echo "----------------------------------------"
        
        $PYTHON_CMD src/train.py \
            --data_dir "$DATA_DIR" \
            --dataset_name "$DATASET" \
            --method "$METHOD" \
            --epochs 50 \
            --batch_size 32 \
            --output_dir "experiments"
            
        echo "Finished $METHOD on $DATASET"
    done
done

echo "All experiments completed."
