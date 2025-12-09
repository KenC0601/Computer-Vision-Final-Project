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
