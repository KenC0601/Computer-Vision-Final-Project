#!/bin/bash

# Activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Warning: .venv not found. Ensure dependencies are installed."
fi

export PYTHONPATH=.

echo "========================================"
echo "Data Download and Setup Script"
echo "========================================"

# 1. Plankton Dataset
echo "----------------------------------------"
echo "1. Downloading/Setting up Plankton Dataset..."
echo "----------------------------------------"
if [ -f "data_download/scripts/download_plankton.py" ]; then
    python data_download/scripts/download_plankton.py
else
    echo "Error: data_download/scripts/download_plankton.py not found."
fi

# 2. Insects 2 Dataset
echo "----------------------------------------"
echo "2. Organizing Insects 2 (IP102) Dataset..."
echo "----------------------------------------"
echo "Note: This step assumes you have the IP102 source data or OpenML access configured."
if [ -f "data_download/scripts/organize_insects2.py" ]; then
    python data_download/scripts/organize_insects2.py
else
    echo "Error: data_download/scripts/organize_insects2.py not found."
fi

echo "========================================"
echo "Data setup process finished."
echo "========================================"
