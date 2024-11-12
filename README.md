# ST-deconv

## 项目简介 | Project Overview
ST-deconv 是一个用于分析空间转录组数据的工具包，包含多个脚本和模型文件，用于模拟空间转录组数据、模型训练及结果分析。

ST-deconv is a toolkit for spatial transcriptomics data analysis, including scripts and model files for data simulation, model training, and result analysis.

## 环境配置 | Environment Setup
项目依赖于 `conda` 环境，您可以下载并使用 `yml` 文件来创建该环境。

This project depends on the `conda` environment. You can download the `yml` file to set up the required environment.

### Conda 环境配置 | Conda Environment Setup
1. 下载 `ST-deconv-environment.yml` 文件。  
   Download the `ST-deconv-environment.yml` file.

2. 在命令行中执行以下命令来创建环境：  
   Run the following command in the terminal to create the environment:

   ```bash
   conda env create -f ST-deconv-environment.yml

3. 激活环境:
   Activate the environment:
   ```bash
   conda activate ST-deconv

### 快速开始 | Quick Start

1.	在 `options.py` 等处修改您的计算机相关路径设置。
Modify your computer’s file path in `options.py` and other relevant files.
2.	运行主脚本 `run_scripts.py` 来执行项目的主流程：
Run the main script `run_scripts.py` to execute the project workflow:
      ```bash 
      python -m ST-deconv.run.run_scripts.py

### 目录结构 | Directory Structure
```bash
ST-deconv
├── data
│   ├── MOB                # 实验使用的 MOB 数据 | MOB data used in the experiment
│   └── experiment         # 结果存储文件夹 | Folder to store results
├── model
│   ├── DANN.py            # Domain adversarial 模型 | Domain adversarial model
│   ├── mix_cell.py        # 模拟空间转录组数据 | Simulate spatial transcriptomic data
│   ├── model.py           # 编码器模型 | Encoder model
│   └── model_nommd.py     # 不含 mmd_loss 的模型 | Model without mmd_loss
├── run
│   ├── finally_test.py    # 空间转录组数据测试 | Spatial transcriptomics data test
│   ├── other_dataset_test.py # 升级版空间转录组数据测试 | Advanced spatial transcriptomics data test
│   ├── run_scripts.py     # 总执行代码 | Main execution script
│   ├── simu_data.py       # 模拟空间转录组数据 | Simulate spatial transcriptomic data
│   └── train.py           # 训练模型 | Train model
├── utils
│   ├── cluster_analyze.py # 模拟时使用的聚类分析 | Clustering analysis used in simulation
│   ├── result_analyze.py  # 结果数据统计 | Result data analysis
│   ├── result_analyze01.py # 结果数据统计（扩展）| Extended result data analysis
│   └── utils.py           # 通用工具函数 | Utility functions
└── options.py             # 各种环境变量 | Configuration for environment variables
└── dataset_config.py      # 数据格式简单处理 | Basic data formatting
