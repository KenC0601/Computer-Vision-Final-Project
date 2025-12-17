import openml
import pandas as pd
import os
import shutil
import sys
from pathlib import Path

# Configuration
DATASET_ID = 44326
PROJECT_ROOT = Path("/home/ken/Computer-Vision-Final-Project")
SOURCE_ROOT = PROJECT_ROOT / "ip102_v1.1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "insects2"
METADATA_DIR = PROJECT_ROOT / "data_download" / "metadata"
METADATA_FILE = METADATA_DIR / "insects_metadata.csv"

# Ensure metadata directory exists
if not os.path.exists(METADATA_DIR):
    os.makedirs(METADATA_DIR)

def download_metadata():
    print(f"Downloading metadata for dataset {DATASET_ID}...")
    try:
        dataset = openml.datasets.get_dataset(DATASET_ID)
        print(f"Dataset Name: {dataset.name}")

        print("Processing metadata...")
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format="dataframe"
        )
        # Combine features and target
        df = pd.concat([X, y], axis=1)
        df.to_csv(METADATA_FILE, index=False)
        print(f"Saved metadata to {METADATA_FILE}")
        return df
    except Exception as e:
        print(f"Error downloading metadata: {e}")
        return None

def organize_images(df, source_dir):
    print(f"Organizing images from {source_dir} into {OUTPUT_DIR}...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Create category folders
    categories = df['CATEGORY'].unique()
    for cat in categories:
        os.makedirs(os.path.join(OUTPUT_DIR, str(cat)), exist_ok=True)
        
    success_count = 0
    missing_count = 0
    
    for _, row in df.iterrows():
        filename = row['FILE_NAME']
        category = row['CATEGORY']
        
        # Source file path (assuming flat structure in source_dir or subfolders)
        src_path = os.path.join(source_dir, filename)
        
        # If not found, try to find it recursively (slow but robust)
        if not os.path.exists(src_path):
             # Try looking in 'images' subfolder if it exists
             src_path_alt = os.path.join(source_dir, 'images', filename)
             if os.path.exists(src_path_alt):
                 src_path = src_path_alt
        
        dst_path = os.path.join(OUTPUT_DIR, str(category), filename)
        
        if os.path.exists(src_path):
            # Copy file
            shutil.copy2(src_path, dst_path)
            success_count += 1
            if success_count % 1000 == 0:
                print(f"Processed {success_count} images...")
        else:
            missing_count += 1
            # print(f"Missing: {filename}") # Too much noise
            
    print(f"Done. Organized {success_count} images. Missing {missing_count} images.")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Organize Insects 2 dataset images.")
    parser.add_argument("--source", help="Path to the source directory containing the images (e.g., IP102_v1.1 folder)", default=None)
    args = parser.parse_args()

    # 1. Download Metadata
    if os.path.exists(METADATA_FILE):
        print(f"Loading existing metadata from {METADATA_FILE}...")
        df = pd.read_csv(METADATA_FILE)
    else:
        df = download_metadata()
        
    if df is None:
        return

    # 2. Check for images
    if args.source:
        source_image_dir = args.source
    else:
        # Expecting images to be placed in data/raw_insects_download
        source_image_dir = os.path.join(os.path.dirname(__file__), "../data/raw_insects_download")
    
    if not os.path.exists(source_image_dir):
        print("\n" + "="*60)
        print("IMPORTANT: IMAGE DOWNLOAD REQUIRED")
        print("="*60)
        print(f"This dataset ({DATASET_ID}) does not allow automatic image downloading via OpenML.")
        print("You need to download the images manually from the official source.")
        print("\n1. Go to: https://github.com/xpwu95/IP102")
        print("2. Look for the 'Data Download' section (Google Drive or AliyunDrive).")
        print("3. Download the 'IP102_v1.1.zip' or similar file.")
        print(f"4. Extract the contents into: {os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw_insects_download'))}")
        print(f"   Ensure the .jpg files are inside that folder (or in an 'images' subfolder within it).")
        print("\nOnce you have done this, run this script again to organize the images.")
        print("="*60)
    else:
        organize_images(df, source_image_dir)

if __name__ == "__main__":
    main()
