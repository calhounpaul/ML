#!/bin/bash

# Source the conda.sh script to ensure conda commands are available
source /opt/conda/etc/profile.d/conda.sh

# Activate the pdf2audio environment
conda activate pdf2audio

# Check if activation was successful
if [[ $CONDA_DEFAULT_ENV != "pdf2audio" ]]; then
    echo "Failed to activate pdf2audio environment"
    exit 1
fi

python /PDF2Audio/app.py
