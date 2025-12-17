# Dataset Download Scripts

This project contains scripts to download and organize two datasets:
1.  **Meta_Album_PLK_Extended (Plankton)** - ID 44317
2.  **Meta_Album_INS_2_Extended (Insects 2)** - ID 44326

## Setup

1.  Ensure you have Python 3 installed.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 1. Downloading Plankton Dataset

This dataset can be downloaded automatically.

1.  Run the script:
    ```bash
    python3 scripts/download_plankton.py
    ```
2.  **Metadata Download**: The script will first check for `data_download/metadata/plankton_metadata.csv`. If missing, it will automatically download the metadata from **OpenML (Dataset ID 44317)**.
3.  **Image Download**: It will then download the images from the WHOI server and save them to `data/plankton/`, organized by category (species).

## 2. Downloading Insects 2 Dataset

This dataset requires manual download of the images first.

1.  **Download Images**:
    *   Go to [https://github.com/xpwu95/IP102](https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo).
    *   Download the dataset (e.g., `IP102_v1.1.zip`) from the provided links.
    *   Extract the zip file.
    *   Move the extracted folder (containing the images) to `data/raw_insects_download`.
        *   Structure should be `data/raw_insects_download/images/00001.jpg` OR `data/raw_insects_download/00001.jpg`.

2.  **Organize Images**:
    *   Run the script:
        ```bash
        python3 scripts/organize_insects2.py
        ```
    *   **Metadata Download**: The script will automatically download the metadata from **OpenML (Dataset ID 44326)** to `data_download/metadata/insects_metadata.csv`.
    *   **Organization**: It will then use this metadata to copy and organize the images into `data/insects2/`, sorted by category.
    *   **Note:** You can also specify a custom source directory:
        ```bash
        python3 scripts/organize_insects2.py --source /path/to/downloaded/folder
        ```


## Folder Structure

*   `scripts/`: Python scripts for downloading and organizing.
*   `metadata/`: CSV files containing dataset metadata (filenames and labels).
*   `data/`: Destination for downloaded images.
