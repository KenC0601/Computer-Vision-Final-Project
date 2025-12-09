# Five-Shot Adaptation of BioCLIP 2 Using LoRA and FlyLoRA
## A Comparative Study on Plankton and Insects 2

**Authors:** Eric Li, Ken Chen

## Project Introduction
BioCLIP 2 is a large biological vision-language foundation model trained on 214 million organism images. This project investigates whether **parameter-efficient fine-tuning (PEFT)** can improve BioCLIP 2's performance in the **five-shot learning** setting on two challenging benchmarks: **Plankton** and **Insects 2** from Meta-Album.

We compare:
1.  A frozen **Linear Probe** baseline.
2.  Standard **LoRA** (Low-Rank Adaptation).
3.  **FlyLoRA**, which introduces an implicit rank-wise mixture-of-experts via sparse random projections.

## Project Structure
```
config/             # Configuration files
data/               # Dataset storage
    splits/         # Train/Val/Test split indices
experiments/        # Experiment logs and checkpoints
    insects2/
    plankton/
report/             # Report assets
src/                # Source code
```

## Datasets
*   **Plankton (Meta-Album PLK):** Marine plankton microscopy images.
*   **Insects 2 (Meta-Album INS_2):** Fine-grained insect photographs.

## Protocol
*   **5-shot training:** 5 labeled images per class.
*   **5-shot validation:** 5 validation images per class.
*   **Test:** All remaining images.

## Methods
*   **Linear Probe:** Frozen encoder + trainable linear classifier.
*   **LoRA:** Low-rank adapters in ViT attention projections.
*   **FlyLoRA:** Rank-wise experts with frozen sparse random projection router.

## Setup
Run `./setup_env.sh` to create the environment and install dependencies.

## Data Preparation
See `README_DATA.md` for instructions on downloading the Plankton and Insects 2 datasets.
For testing, you can generate dummy data:
```bash
python create_dummy_data.py
```

## Running Experiments
Run the experiment script to train all models on all datasets:
```bash
./run_experiments.sh
```
Or run individual experiments:
```bash
python src/train.py --data_dir data/plankton --dataset_name plankton --method flylora
```
