from ..dataset_config import Dataset
from ..options import get_option_list
from ..utils.utils import get_tf_index, hilbert_curve, calculate_rmse, get_newly_filename, reorder_and_fill_missing_vars, create_five_split_testsets
from ..model.model import AEDataset, trainAE, evalAE
from ..model.DANN import trainDAN, DANDataset
import pandas as pd
import numpy as np
import scanpy as sc
import os
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
import os
import ast
from sklearn.model_selection import KFold
import os



# Get option list / initialize dataset
# option = get_option_list()
# Get override options from environment variable
override_options = os.getenv('OVERRIDE_OPTIONS')
option = ast.literal_eval(override_options)  # Parse string into dictionary
dataset = Dataset(option)

# Create directory for storing experiment results
experiment_path = f'{dataset.save_results_dir}'
os.makedirs(experiment_path, exist_ok=True)

dirs_to_create = [
    os.path.join(experiment_path, 'AE_model/Model_parameters/'),
    os.path.join(experiment_path, 'AE_model/Model_result/'),
    os.path.join(experiment_path, 'DANN_model/Model_parameters/'),
]
for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

# Get the path to preprocessed anndata files
if dataset.bool_simu == True:   # ST-deconv simulated spatial data file path
    file_path = f'{dataset.save_results_dir}{dataset.ST_deconv_simu_process_dir}conjustion_data/'

else :  # ST-deconv alternative simulated data file path
    file_path = f'{dataset.save_results_dir}Othersimu/conjustion_data/'

# Load preprocessed anndata files
real_adata = sc.read_h5ad(f'{file_path}adata_real.h5ad')
simu_adata = sc.read_h5ad(f'{file_path}adata_simu.h5ad')


recordered_real_adata = real_adata
recordered_simu_adata = simu_adata


# Create dataloader for data_real
data_real = AEDataset(recordered_real_adata.X, dataset.celltype_list, recordered_real_adata.obs)
data_real_dan = DANDataset(recordered_real_adata.X, dataset.celltype_list)

# Five-fold cross-validation
if dataset.bool_fiveTest == True:
    adata_train, adata_test  = create_five_split_testsets(recordered_simu_adata, current_fold= dataset.fiveTest_fold)

# Define dataloader for data_sim
data_sim = AEDataset(adata_train.X, dataset.celltype_list, adata_train.obs)
data_sim_dan = DANDataset(adata_train.X, dataset.celltype_list)


# Alternating AE-DAN training
for epoch in range(dataset.AE_DAN_epoch):
    trainAE(data_sim, data_real, dataset.AE_batch_size, None, dataset.AE_epochs, experiment_path, dataset.celltype_list, dataset.AE_learning_rate, dataset.bool_DAN, dataset.bool_cl)
    if dataset.bool_DAN:  
        trainDAN(data_sim_dan, data_real_dan, dataset.DAN_batch_size, None, dataset.DAN_epochs, experiment_path, dataset.DAN_learning_rate)

#-------------------------------Testing----------------------------------------
    # Split test
    data_test = AEDataset(adata_test.X, dataset.celltype_list, adata_test.obs)
    prediction = evalAE(data_test, adata_test.shape[0], experiment_path, dataset.celltype_list, 'split_name').values

    ratio = adata_test.obs[dataset.celltype_list].values
    split_rmse = calculate_rmse(prediction, ratio)
    print('split_rmse:', split_rmse)

    # Save RMSE
    results_dir = f'{dataset.save_results_dir}'  
    result_file = os.path.join(results_dir, 'result_rmse.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(result_file, 'a') as file:
        file.write(f'rmse: {split_rmse}\n')

    # External test
    if dataset.other_test_dataset_bool == '1' :
        test_adata = dataset.load_test_data()
        # Align var_names of test data with training data
        test_adata = reorder_and_fill_missing_vars(test_adata, real_adata)

        data_test = AEDataset(test_adata.X, dataset.celltype_list, test_adata.obs)
        prediction = evalAE(data_test, dataset.other_test_dataset_batch_size, experiment_path, dataset.celltype_list, 'card_simu_testdata').values
        
        ratio = test_adata.obs[dataset.celltype_list].values
        rmse = calculate_rmse(prediction, ratio)
        print('rmse:', rmse)

        # Save RMSE
        results_dir = f'{dataset.save_results_dir}'  
        result_file = os.path.join(results_dir, 'result_other_rmse.txt')
        os.makedirs(results_dir, exist_ok=True)
        with open(result_file, 'a') as file:
            file.write(f'othertest_rmse: {rmse}\n')



