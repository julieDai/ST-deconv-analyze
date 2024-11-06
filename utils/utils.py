from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import os
import scanpy as sc
import anndata as ad
from ..utils.cluster_analysis import generate_ref_matrix_location_info
import scipy.sparse as sp

def normalize_sparse_matrix(sparse_matrix):
    # Convert the sparse matrix to CSR format
    sparse_matrix = sparse_matrix.tocsr()

    # Find the maximum value among non-zero elements
    max_value = sparse_matrix.data.max()

    # Scale the non-zero elements to the range [0, 1]
    normalized_matrix = sparse_matrix.multiply(1.0 / max_value)

    return normalized_matrix

def find_obsNames_position(adata, obsNames, filepath):
    # 使用 NumPy 处理数据
    split_data = np.array([name.split('x') for name in obsNames])

    # 检查每个拆分后的数据长度是否为2
    valid_mask = np.array([len(parts) == 2 for parts in split_data])
    valid_data = split_data[valid_mask]

    if not np.all(valid_mask):
        invalid_data = split_data[~valid_mask]
        print(f"Skipping invalid obs_names: {invalid_data}")

    # 将 valid_data 转换为浮点数
    x = valid_data[:, 0].astype(float)
    y = valid_data[:, 1].astype(float)

    # 添加到 adata.obs
    adata.obs['x'] = x
    adata.obs['y'] = y

    # 创建一个 pandas DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    })

    # 将结果保存到指定的文件路径，无索引，文件名为 data_pca_neighbor.csv
    df.to_csv(f"{filepath}/data_pca_neighbor.csv", index=False)

    return adata


def merge_csr_matrices1(matrix1, matrix2, real_column_names_matrix1, real_column_names_matrix2, ratio, file_directory, is_ST_simu):
    # 确保目录存在
    os.makedirs(f'{file_directory}/conjustion_data', exist_ok=True)
    if is_ST_simu== False:
        obsNames = matrix1.obs_names
        matrix1 = matrix1.X
        
    obsNames_real = matrix2.obs_names
    matrix2 = matrix2.X
    # 将输入矩阵转换为CSR格式
    matrix1 = csr_matrix(matrix1)
    matrix2 = csr_matrix(matrix2)

    # 将矩阵转换为DataFrame
    df1 = pd.DataFrame(matrix1.toarray(), columns=real_column_names_matrix1)
    df2 = pd.DataFrame(matrix2.toarray(), columns=real_column_names_matrix2)


    # 计算两个DataFrame列名的并集，并保留列的原始顺序
    all_real_column_names = list(dict.fromkeys(real_column_names_matrix1.tolist() + real_column_names_matrix2.tolist()))
    np.save(f'{file_directory}/conjustion_data/conjustion_gene_names.npy', all_real_column_names)

    # 重建DataFrame，确保包含并集中的所有列，缺失的列用0填充
    df1 = df1.reindex(columns=all_real_column_names, fill_value=0)
    df2 = df2.reindex(columns=all_real_column_names, fill_value=0)

    # 合并DataFrame并立即转换回CSR格式
    simu_merged_matrix = csr_matrix(df1.values)
    real_merged_matrix = csr_matrix(df2.values)

    df_var = pd.DataFrame(index=all_real_column_names)
    adata_simu = ad.AnnData(X=simu_merged_matrix, var=df_var)
    adata_real = ad.AnnData(X=real_merged_matrix, var=df_var)

    # 规格化当前训练数据的位置信息
    adata_real = find_obsNames_position(adata_real, obsNames_real, f'{file_directory}/conjustion_data') 

    if is_ST_simu:
        adata_simu = generate_ref_matrix_location_info(adata_simu, f'{file_directory}/conjustion_data')
    else:
        adata_simu = find_obsNames_position(adata_simu, obsNames, f'{file_directory}/conjustion_data')

    

    # 当前训练数据的ratio信息存入obs
    if type(ratio) == list:
        ratio = pd.DataFrame(ratio)
    else:
        ratio = pd.DataFrame(data = ratio.values, columns = ratio.columns)
    df_obs = pd.DataFrame(adata_simu.obs.values, columns= adata_simu.obs.columns)
    df_combined = ratio.join(df_obs)
    adata_simu.obs = df_combined

    # 对稀疏矩阵进行归一化
    adata_simu.X = normalize_sparse_matrix(adata_simu.X)
    adata_real.X = normalize_sparse_matrix(adata_real.X)

    # 保存归一化的稀疏矩阵
    adata_simu.write(f'{file_directory}conjustion_data/adata_simu.h5ad')
    adata_real.write(f'{file_directory}conjustion_data/adata_real.h5ad')

    # 函数返回adata
    return adata_simu, adata_real


def filter_adata_by_genes(adata_file_path, genes_file_path, output_file_path):
    """
    根据基因列表筛选AnnData对象并保存到新的文件。

    参数:
    adata_file_path (str): 输入AnnData对象的文件路径（.h5ad文件）。
    genes_file_path (str): 包含基因名称的CSV文件路径。
    output_file_path (str): 输出筛选后的AnnData对象的文件路径（.h5ad文件）。
    """
    # 读取AnnData对象
    adata = sc.read_h5ad(adata_file_path)
    
    # 读取基因名称列表
    marker_gene = pd.read_csv(genes_file_path)
    marker_gene_names = marker_gene['GeneName'].unique()

    # 根据筛选后的var_name提取数据
    filtered_adata = adata[:, adata.var_names.isin(marker_gene_names)]
    
    # 保存筛选后的AnnData对象
    filtered_adata.write(output_file_path)
    print(f"Filtered AnnData object saved to {output_file_path}")

def filter_adata_by_genes_data(adata, genes_file_path):
    """
    根据基因列表筛选AnnData对象并保存新的数据。

    参数:
    adata: 输入AnnData对象
    genes_file_path (str): 包含基因名称的CSV文件路径。
    """
    # 读取AnnData对象
    adata = adata
    
    # 读取基因名称列表
    marker_gene = pd.read_csv(genes_file_path)
    marker_gene_names = marker_gene['GeneName'].unique()

    # 根据筛选后的var_name提取数据
    filtered_adata = adata[:, adata.var_names.isin(marker_gene_names)]
    
    return filtered_adata

def hilbert_curve(x, y, level, side, rotation):
    """
    Applies a Hilbert curve mapping to the data points defined by x and y,
    sorts the data points based on their Hilbert curve mapping values, and reorders
    the original 'simu_adata' dataset accordingly.

    Example:
    ```
    # Apply Hilbert curve mapping on each row based on 'x' and 'y' coordinates
    df_simu_adata_position['hilbert_mapping'] = df_simu_adata_position.apply(
        lambda row: hilbert_curve(row['x'], row['y'], 5, 1200, 0), axis=1
    )

    # Sort indices based on the 'hilbert_mapping' values
    sorted_indices_simu = np.argsort(df_simu_adata_position['hilbert_mapping'])

    # Reorder 'simu_adata' based on the sorted indices
    reordered_simu_adata = simu_adata[sorted_indices_simu, :]
    ```
    """

    if level == 0:
        return x, y

    side /= 2

    if rotation == 0:
        x, y = y, x
        x = x + side
    elif rotation == 1:
        x = x + side
        y = y + side
    elif rotation == 2:
        x = side - 1 - x + 2 * side
        y = side - 1 - y + 2 * side
    elif rotation == 3:
        x, y = side - 1 - y + 2 * side, side - 1 - x + 2 * side

    x, y = hilbert_curve(x, y, level - 1, side, (rotation + 1) % 4)
    x, y = hilbert_curve(x, y, level - 1, side, rotation)
    x, y = hilbert_curve(x, y, level - 1, side, rotation)
    x, y = hilbert_curve(x, y, level - 1, side, (rotation + 3) % 4)

    return x, y

from sklearn.neighbors import KDTree, BallTree
def get_tf_index(row_col_data, k):

    # 构建KDTree
    kdtree = KDTree(row_col_data)

    # 构建BallTree
    balltree = BallTree(row_col_data)

    # 获取每一行数据的最近的k-1个点和最远的k-1个点的索引
    closest_indices = kdtree.query(row_col_data, k=k, return_distance=False)[:, 1:]
    farthest_indices = balltree.query(row_col_data, k=k, return_distance=False)[:, 1:]

    return closest_indices,farthest_indices

import os
def get_unique_filename(folder, filename):
    base_name, extension = os.path.splitext(filename)
    unique_name = filename
    counter = 1

    while os.path.exists(os.path.join(folder, unique_name)):
        unique_name = f"{base_name}_{counter}{extension}"
        counter += 1

    return os.path.join(folder, unique_name)

def get_newly_filename(folder, filename):
    base_name, extension = os.path.splitext(filename)
    unique_name = filename
    counter = 1

    while os.path.exists(os.path.join(folder, unique_name)):
        unique_name = f"{base_name}_{counter}{extension}"
        counter += 1

    counter = counter-2
    if counter == 0:
        unique_name = f"{base_name}{extension}"
    else:
        unique_name = f'{base_name}_{counter}{extension}'

    return os.path.join(folder, unique_name)

def get_newly_filename_index(folder, filename):
    base_name, extension = os.path.splitext(filename)
    unique_name = filename
    counter = 1

    while os.path.exists(os.path.join(folder, unique_name)):
        unique_name = f"{base_name}_{counter}{extension}"
        counter += 1

    counter = counter-2
    if counter == 0:
        unique_name = f"{base_name}{extension}"
    else:
        unique_name = f'{base_name}_{counter}{extension}'

    return counter

import math
def calculate_rmse(predictions, targets):
    n = len(predictions)  # 样本数量
    k = len(predictions[0])  # 预测值和真实值的数量

    squared_diff_sum = 0.0  # 平方差总和

    for i in range(n):
        for j in range(k):
            squared_diff = (targets[i][j] - predictions[i][j]) ** 2
            squared_diff_sum += squared_diff

    mean_squared_error = squared_diff_sum / (n * k)  # 平均平方差
    rmse = math.sqrt(mean_squared_error)  # 均方根误差

    return rmse

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
    
    # 对稀疏矩阵进行归一化
    test_adata.X = normalize_sparse_matrix(test_adata.X)

    return test_adata

from sklearn.model_selection import KFold
import numpy as np

def create_five_split_testsets(data, current_fold):
    """
    Splits data into training and testing sets based on the specified fold number using 5-fold cross-validation.

    Parameters:
    data (np.array): The dataset to be split.
    current_fold (int): The fold number (0-4) to select as the current split for testing.

    Returns:
    tuple: A tuple containing two AnnData objects, one for the training set and one for the testing set.

    Example:
    ```
    # Assume 'recordered_simu_adata' is your dataset
    train_adata, test_adata = create_train_test_sets(recordered_simu_adata, current_fold=0)
    ```
    """
    # 创建 5 折交叉验证的分割器
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 获取分割后的索引
    splits = list(kf.split(np.arange(data.shape[0])))

    # 获取当前折叠的训练和测试索引
    train_indices, test_indices = splits[current_fold]

    # 创建训练集和测试集的 AnnData 对象
    adata_train = data[train_indices].copy()
    adata_test = data[test_indices].copy()

    # 打印数据集大小
    print("测试集大小adata_test:", adata_test.shape)
    print("训练集大小adata_train:", adata_train.shape)

    return adata_train, adata_test
