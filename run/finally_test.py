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
    # 获取 real_adata 的 var_names
    real_var_names = real_adata.var_names

    # 初始化一个新的矩阵，大小为 test_adata 的 obs 数量 x real_adata 的 var 数量
    new_X = np.zeros((test_adata.shape[0], len(real_var_names)))

    # 填充已有的 var 数据
    for i, var in enumerate(real_var_names):
        if var in test_adata.var_names:
            new_X[:, i] = test_adata[:, var].X.toarray().flatten()

    # 更新 test_adata 的 X 和 var
    test_adata = ad.AnnData(X=sp.csr_matrix(new_X), 
                            var=pd.DataFrame(index=real_var_names), 
                            obs=test_adata.obs, 
                            obsm=test_adata.obsm, 
                            obsp=test_adata.obsp, 
                            varm=test_adata.varm, 
                            varp=test_adata.varp, 
                            uns=test_adata.uns)

    return test_adata

# 获取选项列表\初始化数据集
# option = get_option_list()
# 获取环境变量中的 override 选项
override_options = os.getenv('OVERRIDE_OPTIONS')
option = ast.literal_eval(override_options)  # 将字符串解析为字典
dataset = Dataset(option)
# dataset.save_results_dir = '/data/ST-deconv/experiment/muti_trainset/test_new_dataset_02/AE/'

# 判断是否使用ST-deconv进行模拟/获取位置信息
if dataset.bool_simu == True:
    file_path = f'{dataset.save_results_dir}{dataset.ST_deconv_simu_process_dir}conjustion_data/'
else :
    file_path = f'{dataset.save_results_dir}Othersimu/conjustion_data/'

real_adata = sc.read_h5ad(f'{file_path}adata_real.h5ad')

# 判断是否使用外来测试数据
if dataset.other_test_dataset_name != 'NULL':
    test_adata = dataset.load_test_data()
    test_adata = reorder_and_fill_missing_vars(test_adata, real_adata)
    experiment_path = f'{dataset.save_results_dir}'
    data_test = AEDataset(test_adata.X, dataset.celltype_list)

    
    evalAE(data_test, dataset.other_test_dataset_batch_size, experiment_path, dataset.celltype_list)
    # 计算rmse
    pre_file_path = get_newly_filename(f'{experiment_path}AE_model/Model_result/', 'predictor_matrix_ref.csv')
    prediction = pd.read_csv(f'{pre_file_path}').values
    ratio = test_adata.obs[dataset.celltype_list].values
    rmse = calculate_rmse(prediction, ratio)
    print('rmse:', rmse)
    # 存储rmse
    results_dir = f'{dataset.save_results_dir}'  
    result_file = os.path.join(results_dir, 'result_rmse.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(result_file, 'w') as file:
        file.write(f'rmse: {rmse}\n')


    



