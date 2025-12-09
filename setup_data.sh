#!/bin/bash

# Setup Data Script
# This script helps set up the Plankton and Insects 2 datasets.
# Due to OpenML API restrictions/network issues in this environment, 
# automated download might fail. This script guides you through the process.

DATA_DIR="data"
mkdir -p "$DATA_DIR"

# Function to check and setup a dataset
setup_dataset() {
    NAME=$1
    ID=$2
    URL=$3
    
    TARGET_DIR="$DATA_DIR/$NAME"
    
    if [ -d "$TARGET_DIR" ] && [ "$(ls -A $TARGET_DIR)" ]; then
        echo "✅ $NAME dataset found in $TARGET_DIR."
    elif [ -f "$DATA_DIR/$NAME.parquet" ]; then
        # Check file size (must be > 1MB to be a valid dataset)
        FILE_SIZE=$(du -k "$DATA_DIR/$NAME.parquet" | cut -f1)
        if [ "$FILE_SIZE" -lt 1000 ]; then
            echo "❌ Found $NAME.parquet but it is too small ($FILE_SIZE KB). It might be an error file."
            echo "   Please delete it and download the correct file manually."
            rm "$DATA_DIR/$NAME.parquet"
        else
            echo "📦 Found $NAME.parquet. Extracting..."
            python src/extract_parquet.py --file "$DATA_DIR/$NAME.parquet" --output "$TARGET_DIR"
            if [ -d "$TARGET_DIR" ] && [ "$(ls -A $TARGET_DIR)" ]; then
                 echo "✅ $NAME dataset extracted successfully."
            else
                 echo "❌ Extraction failed. Please check the parquet file."
            fi
        fi
    else
        echo "----------------------------------------------------------------"
        echo "⚠️  $NAME dataset not found or empty."
        echo "Please download the dataset manually."
        echo ""
        echo "Dataset: $NAME (Meta-Album Mini)"
        echo "OpenML Page: https://www.openml.org/d/$ID"
        echo "Direct Parquet Download: https://openml.org/datasets/0004/$ID/dataset_$ID.pq"
        echo ""
        echo "⚠️  CRITICAL ISSUE: The OpenML download links for the Parquet files appear to be broken"
        echo "    (they redirect to 'localhost' and fail). This affects both the API and direct downloads."
        echo ""
        echo "Instructions:"
        echo "1. Try to download the Parquet file from the link above on a different network/machine."
        echo "   If it works, move it to '$DATA_DIR/$NAME.parquet'."
        echo "2. If you cannot download the real data, please use the dummy data generator to test the code:"
        echo "   python create_dummy_data.py"
        echo ""
        echo "3. If you have the file, run this script again to extract."
        echo "----------------------------------------------------------------"
    fi
}

# Create extraction script just in case user gets parquet
cat > src/extract_parquet.py <<EOF
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import sys

def extract(parquet_file, output_dir):
    print(f"Reading {parquet_file}...")
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Error: {e}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find pixel columns
    pixel_cols = [c for c in df.columns if str(c).isdigit()]
    pixel_cols.sort(key=int)
    
    if not pixel_cols:
        print("No pixel columns found in parquet.")
        return

    print(f"Extracting {len(df)} images to {output_dir}...")
    for idx, row in df.iterrows():
        # Try to find label
        label = "unknown"
        for col in ['category', 'label', 'class', 'CATEGORY']:
            if col in df.columns:
                label = str(row[col])
                break
        
        class_dir = output_dir / label
        class_dir.mkdir(exist_ok=True)
        
        try:
            pixels = row[pixel_cols].values.astype(np.uint8)
            if len(pixels) == 128*128*3:
                img = Image.fromarray(pixels.reshape(128, 128, 3))
                img.save(class_dir / f"img_{idx}.png")
        except Exception as e:
            pass
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    extract(args.file, args.output)
EOF

echo "Checking datasets..."
setup_dataset "plankton" "44282" "https://www.openml.org/d/44282"
setup_dataset "insects2" "44292" "https://www.openml.org/d/44292"

echo ""
echo "To generate dummy data for testing instead, run:"
echo "python create_dummy_data.py"
