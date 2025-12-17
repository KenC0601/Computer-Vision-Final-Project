# BioCLIP 2 Fine-Tuning on Insects 2 and Plankton
## Full Dataset Fine-Tuning with 5-Shot Evaluation

**Authors:** Eric Li, Ken Chen

## Project Overview
This project explores the effectiveness of **Parameter-Efficient Fine-Tuning (PEFT)** methods on the **BioCLIP 2** foundation model for fine-grained biological classification. We focus on two datasets:
1.  **Insects 2**: A large-scale dataset for insect pest recognition.
2.  **Plankton**: A dataset of marine plankton images (to be added).

We compare five approaches:
1.  **Baseline:** Frozen BioCLIP 2 backbone + 5-shot Linear Probe.
2.  **LoRA (Low-Rank Adaptation):** Standard PEFT method.
3.  **FlyLoRA:** A bio-inspired sparse mixture-of-experts adapter.
4.  **DoRA (Weight-Decomposed LoRA):** Decomposes weights into magnitude and direction for better stability.
5.  **PiSSA (Principal Singular Values Adaptation):** Initializes adapters using SVD of the pre-trained weights.

## Methods

### 1. LoRA (Low-Rank Adaptation)
Injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.

### 2. FlyLoRA
Inspired by the fruit fly's olfactory system, FlyLoRA uses a sparse mixture-of-experts approach. It projects inputs into a high-dimensional sparse representation, allowing for efficient and specialized adaptation.

### 3. DoRA (Weight-Decomposed LoRA)
DoRA decomposes the pre-trained weight matrix into a magnitude vector and a directional matrix. It applies LoRA only to the directional component, which often leads to better learning stability and performance closer to full fine-tuning.

### 4. PiSSA (Principal Singular Values Adaptation)
PiSSA initializes the LoRA matrices based on the principal singular values and vectors of the original model's weights. This provides a "warm start" for the adapters, potentially leading to faster convergence and better results than random initialization.

## Project Structure
```
data/               # Dataset storage
    insects2/       # Insects 2 dataset images
    plankton/       # Plankton dataset images
data_download/      # Scripts for downloading and organizing data
experiments/        # Checkpoints from dataset training
report/             # Jupyter notebooks for evaluation and visualization
    insects2_5shot_eval.ipynb
    plankton_5shot_eval.ipynb
    results_visualization.ipynb
src/                # Source code
    dataset.py      # Data loading and splitting logic
    models.py       # BioCLIP, LoRA, FlyLoRA, DoRA, PiSSA definitions
    train.py        # Main training script
    eval_5shot.py   # 5-shot evaluation script
    peft_utils/     # PEFT method implementations
train_all_insects2.sh  # Script to run full fine-tuning for Insects 2
train_all_plankton.sh  # Script to run full fine-tuning for Plankton
setup_env.sh           # Script to setup environment
setup_data.sh          # Script to set up dataset (READ data_download/README.md before using this script)
```

## Prerequisites
*   **Hardware:** NVIDIA GPU with at least 32GB VRAM.
*   **OS:** Linux (tested on Ubuntu).

## Setup

1.  **Environment Setup**
    Create a virtual environment and install dependencies:
    ```bash
    ./setup_env.sh
    ```

2.  **Data Preparation**
    We provide scripts in `data_download/` to help download and organize the datasets.
    
    *   **Plankton:**
        ```bash
        python data_download/scripts/download_plankton.py
        ```
        This will download metadata from OpenML and images from WHOI.

    *   **Insects 2:**
        First, download the raw images manually (see `data_download/README.md`). Then run:
        ```bash
        python data_download/scripts/organize_insects2.py
        ```
        This will download metadata from OpenML and organize the images.

    Ensure the data is located in `data/insects2` and `data/plankton`.

## Running the Project

### 1. Full Dataset Fine-Tuning
We provide automated shell scripts to train all PEFT methods (LoRA, FlyLoRA, DoRA, PiSSA) sequentially. These scripts handle the specific batch size requirements for each method.

**Run Insects 2 Training (20 Epochs):**
```bash
./train_all_insects2.sh
```
*   LoRA/FlyLoRA: Batch Size 100
*   DoRA: Batch Size 60
*   PiSSA: Batch Size 80

**Run Plankton Training (3 Epochs):**
```bash
./train_all_plankton.sh
```
*   LoRA/FlyLoRA: Batch Size 100
*   DoRA: Batch Size 60
*   PiSSA: Batch Size 80

Alternatively, you can run `src/train.py` manually:
```bash
python src/train.py \
  --dataset_name [plankton|insects2] \
  --data_dir data/ [plankton|insects2] \
  --method [lora|flylora|dora|pissa] \
  --epochs [20 (Insects 2)/3 (Plankton)] \
  --batch_size 100
```

**Specific Examples:**

*   **LoRA (Baseline PEFT):**
    ```bash
    python src/train.py --dataset_name plankton --data_dir data/plankton --method lora --epochs 3
    ```

*   **FlyLoRA (Bio-Inspired):**
    ```bash
    python src/train.py --dataset_name plankton --data_dir data/plankton --method flylora --epochs 3
    ```

*   **DoRA (Weight-Decomposed LoRA):**
    ```bash
    python src/train.py --dataset_name plankton --data_dir data/plankton --method dora --epochs 3
    ```

*   **PiSSA (Principal Singular Values Adaptation):**
    ```bash
    python src/train.py --dataset_name plankton --data_dir data/plankton --method pissa --epochs 3
    ```

### 2. 5-Shot Evaluation
We use Jupyter Notebooks for the 5-shot evaluation to allow for interactive analysis and visualization.

1.  **Open the Notebooks:**
    *   For Plankton: `report/plankton_5shot_eval.ipynb`
    *   For Insects 2: `report/insects2_5shot_eval.ipynb`

2.  **Run the Evaluation:**
    *   Open the desired notebook in VS Code or Jupyter Lab.
    *   Run all cells.
    *   The notebook will:
        *   Load the fine-tuned models from the `experiments/` directory.
        *   Perform 5-shot linear probing (training on 5 images per class).
        *   Evaluate on the test set.
        *   Generate a results CSV and summary table.

**Note:** Ensure you have run the training step first so that the model checkpoints (e.g., `experiments/plankton/lora/best_model.pth`) exist. The notebooks are configured to look for these specific paths.

### 3. Visualization
We provide a dedicated notebook to visualize and compare the results across all methods and datasets.

1.  **Open `report/results_visualization.ipynb`**.
2.  Run all cells.
3.  This will generate:
    Accuracy Comparison Bar Chart: Comparing Baseline, LoRA, FlyLoRA, DoRA, and PiSSA.

## Results
The evaluation notebooks generate CSV files containing the detailed results:
*   `insects2_5shot_results.csv`
*   `plankton_5shot_results.csv`

These files are automatically loaded by the visualization notebook to produce the final graphs.