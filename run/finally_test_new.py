from ..dataset_config import Dataset
from ..options import get_option_list, get_base_option_list
from ..model.model import AEDataset, evalAE
from ..utils.utils import hilbert_curve, calculate_rmse, get_newly_filename, find_obsNames_position
import numpy as np
import pandas as pd
import os
import ast
import scanpy as sc
import anndata as ad
import scipy.sparse as sp


# Get base option list / initialize dataset
option = get_base_option_list()

dataset = Dataset(option)
dataset.save_results_dir = f'/home/daishurui/git_project/ST-deconv/data/experiment/CL4.0/trainModel_xiaorong_CLloss*0.01_01'



# Determine whether to use ST-deconv simulation -> Get correct ST-deconv preprocessed dataset path
if dataset.bool_simu == True:
    file_path = f'{dataset.save_results_dir}{dataset.ST_deconv_simu_process_dir}conjustion_data/'
else :
    file_path = f'{dataset.save_results_dir}Othersimu/conjustion_data/'

real_adata = sc.read_h5ad(f'{file_path}adata_real.h5ad')



# Use real spatial transcriptomics data for testing
experiment_path = f'{dataset.save_results_dir}'
real_adata_test = AEDataset(real_adata.X, dataset.celltype_list, real_adata.obs)
    
evalAE(real_adata_test, dataset.other_test_dataset_batch_size, experiment_path, dataset.celltype_list,'true_testdata')

pre_file_path = get_newly_filename(f'{experiment_path}AE_model/Model_result/true_testdata', 'predictor_matrix_ref.csv')


    



