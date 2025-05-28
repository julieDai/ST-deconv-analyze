from ..dataset_config import Dataset
from ..options import get_option_list
from ..model.model import AEDataset, evalAE
from ..utils.utils import hilbert_curve, calculate_rmse, get_newly_filename
import numpy as np
import pandas as pd
import os
import ast
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

def reorder_and_fill_missing_vars(test_adata, real_adata):
    # Get var_names from real_adata
    real_var_names = real_adata.var_names

    # Initialize a new matrix with shape: number of obs in test_adata × number of vars in real_adata
    new_X = np.zeros((test_adata.shape[0], len(real_var_names)))

    # Fill in existing variable data
    for i, var in enumerate(real_var_names):
        if var in test_adata.var_names:
            new_X[:, i] = test_adata[:, var].X.toarray().flatten()

    # Update test_adata's X and var   
    test_adata = ad.AnnData(X=sp.csr_matrix(new_X), 
                            var=pd.DataFrame(index=real_var_names), 
                            obs=test_adata.obs, 
                            obsm=test_adata.obsm, 
                            obsp=test_adata.obsp, 
                            varm=test_adata.varm, 
                            varp=test_adata.varp, 
                            uns=test_adata.uns)

    return test_adata

# Get option list / initialize dataset
# option = get_option_list()
# Get override options from environment variable
override_options = os.getenv('OVERRIDE_OPTIONS')
option = ast.literal_eval(override_options)  # Parse the string to a dictionary
dataset = Dataset(option)
# dataset.save_results_dir = '/data/ST-deconv/experiment/muti_trainset/test_new_dataset_02/AE/'

# Determine whether to use ST-deconv simulation / get spatial info
if dataset.bool_simu == True:
    file_path = f'{dataset.save_results_dir}{dataset.ST_deconv_simu_process_dir}conjustion_data/'
else :
    file_path = f'{dataset.save_results_dir}Othersimu/conjustion_data/'

real_adata = sc.read_h5ad(f'{file_path}adata_real.h5ad')

# Check if external test dataset is used
if dataset.other_test_dataset_name != 'NULL':
    test_adata = dataset.load_test_data()
    test_adata = reorder_and_fill_missing_vars(test_adata, real_adata)
    experiment_path = f'{dataset.save_results_dir}'
    data_test = AEDataset(test_adata.X, dataset.celltype_list)

    
    evalAE(data_test, dataset.other_test_dataset_batch_size, experiment_path, dataset.celltype_list)
    # Calculate RMSE
    pre_file_path = get_newly_filename(f'{experiment_path}AE_model/Model_result/', 'predictor_matrix_ref.csv')
    prediction = pd.read_csv(f'{pre_file_path}').values
    ratio = test_adata.obs[dataset.celltype_list].values
    rmse = calculate_rmse(prediction, ratio)
    print('rmse:', rmse)
    # Save RMSE to file
    results_dir = f'{dataset.save_results_dir}'  
    result_file = os.path.join(results_dir, 'result_rmse.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(result_file, 'w') as file:
        file.write(f'rmse: {rmse}\n')


    



