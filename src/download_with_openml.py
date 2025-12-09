import openml
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# Configure OpenML to use a specific cache directory if needed, 
# but default is usually fine (~/.openml/cache)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_and_extract(dataset_id, name):
    print(f"Attempting to download {name} (ID: {dataset_id}) using OpenML API...")
    try:
        # This will try to download the dataset metadata and data
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        
        print(f"Dataset {name} metadata downloaded.")
        print("Loading data (this might take a while)...")
        
        # Get the data as a pandas DataFrame
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        
        print(f"Data loaded. Shape: {X.shape}")
        
        # Create output directory
        output_dir = DATA_DIR / name
        output_dir.mkdir(exist_ok=True)
        
        # Find pixel columns (assuming they are the numeric ones)
        # Meta-Album datasets usually have pixel columns named '0', '1', ... or similar
        # But let's check the columns
        pixel_cols = [c for c in X.columns if str(c).isdigit()]
        pixel_cols.sort(key=int)
        
        if not pixel_cols:
            print(f"No pixel columns found for {name}. Columns: {X.columns[:10]}...")
            return
            
        print(f"Extracting images to {output_dir}...")
        
        # Combine X and y to iterate easily
        df = X.copy()
        df['label'] = y
        
        for idx, row in df.iterrows():
            label = str(row['label'])
            class_dir = output_dir / label
            class_dir.mkdir(exist_ok=True)
            
            try:
                pixels = row[pixel_cols].values.astype(np.uint8)
                # Assuming 128x128x3 images based on description
                if len(pixels) == 128*128*3:
                    img = Image.fromarray(pixels.reshape(128, 128, 3))
                    img.save(class_dir / f"img_{idx}.png")
                else:
                    # Fallback or error
                    pass
            except Exception as e:
                print(f"Error saving image {idx}: {e}")
                
        print(f"Successfully extracted {name}.")
        
    except Exception as e:
        print(f"FAILED to download {name} using OpenML API.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    # Plankton Mini
    download_and_extract(44282, "plankton")
    print("-" * 50)
    # Insects 2 Mini
    download_and_extract(44292, "insects2")
