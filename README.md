# ST-deconv

## Overview

**ST-deconv** is a toolkit for spatial transcriptomics data analysis. It includes scripts and model files for data simulation, model training, and result analysis.

## Environment Setup

This project uses a `conda` environment. To set it up:

```bash
# Step 1: Download the environment configuration file
# File: ST-deconv-environment.yml

# Step 2: Create the environment
conda env create -f ST-deconv-environment.yml

# Step 3: Activate the environment
conda activate ST-deconv
```

## Quick Start

```bash
# 1. Modify paths in `options.py` and other related config files

# 2. Run the main script to execute the workflow
python -m ST-deconv.run.run_scripts.py
```

## Directory Structure

```bash
ST-deconv
├── data
│   ├── MOB/                     # MOB dataset used in the experiment
│   └── experiment/              # Folder to store experiment results
├── model
│   ├── DANN.py                  # Domain adversarial model
│   ├── mix_cell.py              # Simulate spatial transcriptomics data
│   ├── model.py                 # Encoder model
│   └── model_nommd.py           # Model without MMD loss
├── run
│   ├── finally_test.py          # Test script for spatial transcriptomics data
│   ├── other_dataset_test.py    # Test script for additional datasets
│   ├── run_scripts.py           # Main execution script
│   ├── simu_data.py             # Spatial data simulation script
│   └── train.py                 # Model training script
├── utils
│   ├── cluster_analyze.py       # Clustering analysis functions
│   ├── result_analyze.py        # Result analysis
│   ├── result_analyze01.py      # Extended result analysis
│   └── utils.py                 # Utility functions
├── draw_pic                    # R scripts for data visualization
│   ├── MOB_pic.R                # Visualization for the MOB dataset
│   └── PDAC_pic.R               # Visualization for the PDAC dataset
├── options.py                  # Configuration for dataset paths and parameters
└── dataset_config.py           # Basic dataset formatting script
```

## Dataset Configuration

It is recommended to store all datasets in the `data/` directory, each in a subfolder named after the dataset.

Example:

```bash
data/
├── MOB/
├── DPAC.zip     # Large dataset compressed archive
```

> ⚠️ Before running new dataset in the program, make sure to correctly configure the paths and parameters in `options.py`.

### Configuration File: `options.py`

This file is used to add and manage new spatial transcriptomics datasets, including:

- Path to spatial transcriptomics expression data  
- Path to spatial location data  
- Path to single-cell transcriptomics data  
- Path to single-cell cell type labels  

Refer to the MOB dataset configuration in `options.py` as a template when adding new datasets.

## Data Visualization

- Visualization scripts are located in the `draw_pic/` directory and written in **R**.
- It is recommended to refer to the [CARD project](https://github.com/YingMa0107/CARD) for additional visualization methods and examples.

