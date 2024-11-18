# ST-deconv

## Project Overview
ST-deconv is a toolkit for spatial transcriptomics data analysis, including scripts and model files for data simulation, model training, and result analysis.

## 环境配置
This project depends on the `conda` environment. You can download the `yml` file to set up the required environment.

### Conda Environment Setup
1.   Download the `ST-deconv-environment.yml` file.

2. Run the following command in the terminal to create the environment:

   ```bash
   conda env create -f ST-deconv-environment.yml

3. Activate the environment:
   ```bash
   conda activate ST-deconv

### Quick Start

1.	Modify your computer’s file path in `options.py` and other relevant files.
2.	Run the main script `run_scripts.py` to execute the project workflow:
      ```bash 
      python -m ST-deconv.run.run_scripts.py

### Directory Structure
```bash
ST-deconv
├── data
│   ├── MOB                # MOB data used in the experiment
│   └── experiment         # Folder to store results
├── model
│   ├── DANN.py            # Domain adversarial model
│   ├── mix_cell.py        # Simulate spatial transcriptomic data
│   ├── model.py           # Encoder model
│   └── model_nommd.py     # Model without mmd_loss
├── run
│   ├── finally_test.py    # Spatial transcriptomics data test
│   ├── other_dataset_test.py # Advanced spatial transcriptomics data test
│   ├── run_scripts.py     # Main execution script
│   ├── simu_data.py       # Simulate spatial transcriptomic data
│   └── train.py           # Train model
├── utils
│   ├── cluster_analyze.py # Clustering analysis used in simulation
│   ├── result_analyze.py  # Result data analysis
│   ├── result_analyze01.py # Extended result data analysis
│   └── utils.py           # Utility functions
└── options.py             # Configuration for environment variables
└── dataset_config.py      # Basic data formatting
