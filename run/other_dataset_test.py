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
def normalize_sparse_matrix(sparse_matrix):
    # Convert the sparse matrix to CSR format
    sparse_matrix = sparse_matrix.tocsr()

    # Find the maximum value among non-zero elements
    max_value = sparse_matrix.data.max()

    # Scale the non-zero elements to the range [0, 1]
    normalized_matrix = sparse_matrix.multiply(1.0 / max_value)

    return normalized_matrix

def reorder_and_fill_missing_vars(pd_test_adata, real_adata):
    # 获取 real_adata 的 var_names
    real_var_names = real_adata.var_names

    # 初始化一个新的矩阵，大小为 test_adata 的 obs 数量, x real_adata 的 var 数量
    new_X = np.zeros((pd_test_adata.shape[0], len(real_var_names)))

    # 填充已有的 var 数据
    for i, var in enumerate(real_var_names):
        if var in pd_test_adata.columns:
            new_X[:, i] = pd_test_adata[var].values

    # 更新 test_adata 的 X 和 var
    # obs_df = pd.DataFrame(index=pd_test_adata.index)
    obs_df = real_adata.obs[:-2]
    new_X=sp.csr_matrix(new_X)
    new_X = normalize_sparse_matrix(new_X)
    test_adata = ad.AnnData(X=new_X, 
                            var=pd.DataFrame(index=real_var_names), 
                            obs=obs_df, )

    return test_adata

# 判断是否使用ST-deconv进行模拟->获取正确的ST-deconv预处理好的数据集路径
def get_real_adata(folder_name, dataset):

    if 'simu' in folder_name:
        simu_file_path = f'{folder_name}/{dataset.ST_deconv_simu_process_dir}'
    else:
        simu_file_path = f'{folder_name}/Othersimu/'

    real_adata = sc.read_h5ad(f'{simu_file_path}conjustion_data/adata_real.h5ad')
    return real_adata, simu_file_path



# 获取选项列表\初始化数据集
option = get_base_option_list()

dataset = Dataset(option)
dataset.save_results_dir = f'/home/daishurui/git_project/ST-deconv/data/experiment/CL4.0/trainModel_xiaorong_CLloss*0.01_01'
target_dataset_dir = '/home/daishurui/git_project/ST-deconv/data/MOB/testset/Card_simu_data'
    


for folder_name in os.listdir(dataset.save_results_dir):
    folder_path = os.path.join(dataset.save_results_dir, folder_name)


    # 检查是否为文件夹
    if os.path.isdir(folder_path):
        real_adata, simu_file_path = get_real_adata(folder_path, dataset)
        print(f"Processing folder: {folder_name}")
        real_adata_test = AEDataset(real_adata.X, dataset.celltype_list, real_adata.obs)

        with open(f'{simu_file_path}rmse.txt', 'w') as file:
            file.write('there is card test rmse:')

    # 遍历数据集计算rmse并存入
    for data_folder_name in os.listdir(target_dataset_dir):
        
        data_folder_path = os.path.join(target_dataset_dir, data_folder_name)
        test_data = pd.read_csv(f'{data_folder_path}/pseudo_data_{data_folder_name}.csv', index_col=0, header=0).T
        test_label = pd.read_csv(f'{data_folder_path}/pseudo_data_ratio_{data_folder_name}.csv', header=0)
        test_data_normaliation = reorder_and_fill_missing_vars(test_data, real_adata)
        test_data_normaliation_AEdataset = AEDataset(test_data_normaliation.X, dataset.celltype_list, test_data_normaliation.obs)

        ratio = evalAE(test_data_normaliation_AEdataset, 260, f'{folder_path}/', dataset.celltype_list, f'card_simu_testdata/{data_folder_name}/')
        rmse = calculate_rmse(ratio.T, test_label.T)
        # 追加数据到文件
        with open(f'{simu_file_path}rmse.txt', 'a') as file:
            file.write(f'\n{data_folder_name}:{rmse}')
        



# # 使用真实空间转录组数据测试
# experiment_path = f'{dataset.save_results_dir}'
# real_adata_test = AEDataset(real_adata.X, dataset.celltype_list, real_adata.obs)
    
# evalAE(real_adata_test, dataset.other_test_dataset_batch_size, experiment_path, dataset.celltype_list,'true_testdata')

# pre_file_path = get_newly_filename(f'{experiment_path}AE_model/Model_result/true_testdata', 'predictor_matrix_ref.csv')


    



