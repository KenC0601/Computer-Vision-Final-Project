# Dataset Setup Instructions

The automated download script attempts to download the "Mini" versions of the datasets from OpenML. However, due to potential network issues or OpenML API limitations, you may need to download the datasets manually.

## Manual Download

1.  **Plankton (Meta-Album PLK)**
    *   Visit: [https://meta-album.github.io/datasets/PLK.html](https://meta-album.github.io/datasets/PLK.html)
    *   Download the "Extended" (recommended) or "Mini" dataset zip file.
    *   Extract the contents into `data/plankton`.
    *   Ensure the structure is: `data/plankton/<class_name>/<image_file>`

2.  **Insects 2 (Meta-Album INS_2)**
    *   Visit: [https://meta-album.github.io/datasets/INS_2.html](https://meta-album.github.io/datasets/INS_2.html)
    *   Download the "Extended" or "Mini" dataset zip file.
    *   Extract the contents into `data/insects2`.
    *   Ensure the structure is: `data/insects2/<class_name>/<image_file>`

## Using Dummy Data
For testing the pipeline, you can generate dummy data using:
```bash
python create_dummy_data.py
```
This will create random noise images in `data/plankton` and `data/insects2`.
