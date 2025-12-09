#!/bin/bash

# Install Mambaforge (optional, if not already installed)
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
# bash Mambaforge-Linux-x86_64.sh

# Create environment
mamba create -n bioclip_peft python=3.10 -y

# Activate environment
source activate bioclip_peft

# Install dependencies
pip install -r requirements.txt

echo "Environment setup complete. Activate with 'conda activate bioclip_peft' or 'source activate bioclip_peft'"
