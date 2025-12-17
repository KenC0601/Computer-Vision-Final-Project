from anyio import Path
import pandas as pd
import requests
import os
import time
import openml
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
# Path relative to where the script is run (assuming run from project root or scripts folder)

DATASET_ID = 44317
PROJECT_ROOT = Path("/home/ken/Computer-Vision-Final-Project")
OUTPUT_DIR = PROJECT_ROOT / "data" / "plankton"
METADATA_DIR = PROJECT_ROOT / "data_download" / "metadata"
METADATA_FILE = METADATA_DIR / "plankton_metadata.csv"

# Ensure metadata directory exists
if not os.path.exists(METADATA_DIR):
    os.makedirs(METADATA_DIR)

DOWNLOAD_LIMIT = None  # Set to None to download all images
MAX_WORKERS = 40      # Increased workers for faster checking/downloading

def download_metadata():
    print(f"Downloading metadata for dataset {DATASET_ID} from OpenML...")
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

def get_file_path(row):
    filename = row['FILE_NAME']
    category = row['CATEGORY']
    category_dir = os.path.join(OUTPUT_DIR, str(category))
    return os.path.join(category_dir, filename), category_dir

def download_image(row):
    filename = row['FILE_NAME']
    category = row['CATEGORY']
    file_path, category_dir = get_file_path(row)
    
    # Create category directory if it doesn't exist (thread-safe enough for os.makedirs exist_ok=True)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir, exist_ok=True)
    
    # Double check existence (in case it was created since the pre-check)
    if os.path.exists(file_path):
        return "skipped"
    
    url = f"https://ifcb-data.whoi.edu/mvco/{filename}"
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return "downloaded"
        else:
            return f"failed_status_{response.status_code}"
    except Exception as e:
        return f"error_{str(e)}"

def main():
    if not os.path.exists(METADATA_FILE):
        print(f"{METADATA_FILE} not found. Attempting to download...")
        df = download_metadata()
        if df is None:
            print("Failed to download metadata. Exiting.")
            return
    else:
        print(f"Reading metadata from {METADATA_FILE}...")
        df = pd.read_csv(METADATA_FILE)
    
    if DOWNLOAD_LIMIT:
        print(f"Limiting download to first {DOWNLOAD_LIMIT} images.")
        df = df.head(DOWNLOAD_LIMIT)

    total_images = len(df)
    print(f"Total images in dataset: {total_images}")
    
    # Pre-check for existing files to skip them efficiently
    print("Checking for existing files...")
    rows_to_download = []
    existing_count = 0
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for _, row in df.iterrows():
        file_path, _ = get_file_path(row)
        if os.path.exists(file_path):
            existing_count += 1
        else:
            rows_to_download.append(row)
            
    print(f"Found {existing_count} existing images.")
    print(f"Remaining to download: {len(rows_to_download)}")
    
    if not rows_to_download:
        print("All images already downloaded!")
        return

    print(f"Starting download with {MAX_WORKERS} workers...")
    
    start_time = time.time()
    downloaded_count = 0
    error_count = 0
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(download_image, row): row['FILE_NAME'] for row in rows_to_download}
        
        for i, future in enumerate(as_completed(future_to_file)):
            result = future.result()
            
            if result == "downloaded":
                downloaded_count += 1
            elif result.startswith("failed") or result.startswith("error"):
                error_count += 1
                # Optional: print errors
                # print(f"Error downloading {future_to_file[future]}: {result}")

            # Print progress every 100 images
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"Progress: {i + 1}/{len(rows_to_download)} ({rate:.2f} imgs/sec) - Downloaded: {downloaded_count}, Errors: {error_count}")

    print(f"\nDownload process completed.")
    print(f"New downloads: {downloaded_count}")
    print(f"Errors: {error_count}")
    print(f"Total images available: {existing_count + downloaded_count}/{total_images}")

if __name__ == "__main__":
    main()
