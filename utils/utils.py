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
    # Use NumPy to process the data
    split_data = np.array([name.split('x') for name in obsNames])

    # Check if each split item has exactly 2 parts
    valid_mask = np.array([len(parts) == 2 for parts in split_data])
    valid_data = split_data[valid_mask]

    if not np.all(valid_mask):
        invalid_data = split_data[~valid_mask]
        print(f"Skipping invalid obs_names: {invalid_data}")

    # Convert valid_data to float
    x = valid_data[:, 0].astype(float)
    y = valid_data[:, 1].astype(float)

    # Add to adata.obs
    adata.obs['x'] = x
    adata.obs['y'] = y

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    })

    # Save the results to the specified path, without index, file name: data_pca_neighbor.csv
    df.to_csv(f"{filepath}/data_pca_neighbor.csv", index=False)

    return adata


def merge_csr_matrices1(matrix1, matrix2, real_column_names_matrix1, real_column_names_matrix2, ratio, file_directory, is_ST_simu):
    # Ensure the directory exists
    os.makedirs(f'{file_directory}/conjustion_data', exist_ok=True)
    if is_ST_simu== False:
        obsNames = matrix1.obs_names
        matrix1 = matrix1.X
        
    obsNames_real = matrix2.obs_names
    matrix2 = matrix2.X
    # Convert the input matrices to CSR format
    matrix1 = csr_matrix(matrix1)
    matrix2 = csr_matrix(matrix2)

    # Convert matrices to DataFrames
    df1 = pd.DataFrame(matrix1.toarray(), columns=real_column_names_matrix1)
    df2 = pd.DataFrame(matrix2.toarray(), columns=real_column_names_matrix2)


    # Compute the union of column names from both DataFrames, preserving order
    all_real_column_names = list(dict.fromkeys(real_column_names_matrix1.tolist() + real_column_names_matrix2.tolist()))
    np.save(f'{file_directory}/conjustion_data/conjustion_gene_names.npy', all_real_column_names)

    # Rebuild DataFrames to ensure all columns in the union are present, filling missing values with 0
    df1 = df1.reindex(columns=all_real_column_names, fill_value=0)
    df2 = df2.reindex(columns=all_real_column_names, fill_value=0)

    # Merge the DataFrames and immediately convert them back to CSR format
    simu_merged_matrix = csr_matrix(df1.values)
    real_merged_matrix = csr_matrix(df2.values)

    df_var = pd.DataFrame(index=all_real_column_names)
    adata_simu = ad.AnnData(X=simu_merged_matrix, var=df_var)
    adata_real = ad.AnnData(X=real_merged_matrix, var=df_var)

    # Normalize the location information for the current training data
    adata_real = find_obsNames_position(adata_real, obsNames_real, f'{file_directory}/conjustion_data') 

    if is_ST_simu:
        adata_simu = generate_ref_matrix_location_info(adata_simu, f'{file_directory}/conjustion_data')
    else:
        adata_simu = find_obsNames_position(adata_simu, obsNames, f'{file_directory}/conjustion_data')

    

    # Store the ratio information into obs for the current training data
    if type(ratio) == list:
        ratio = pd.DataFrame(ratio)
    else:
        ratio = pd.DataFrame(data = ratio.values, columns = ratio.columns)
    df_obs = pd.DataFrame(adata_simu.obs.values, columns= adata_simu.obs.columns)
    df_combined = ratio.join(df_obs)
    adata_simu.obs = df_combined

    # Normalize the sparse matrices
    adata_simu.X = normalize_sparse_matrix(adata_simu.X)
    adata_real.X = normalize_sparse_matrix(adata_real.X)

    # Save the normalized sparse matrices
    adata_simu.write(f'{file_directory}conjustion_data/adata_simu.h5ad')
    adata_real.write(f'{file_directory}conjustion_data/adata_real.h5ad')

    # Return the merged AnnData objects
    return adata_simu, adata_real


def filter_adata_by_genes(adata_file_path, genes_file_path, output_file_path):
    """
    Filter an AnnData object based on a list of genes and save to a new file.

    Parameters:
    adata_file_path (str): File path to the input AnnData object (.h5ad file).
    genes_file_path (str): File path to a CSV file containing gene names.
    output_file_path (str): File path to save the filtered AnnData object (.h5ad file).
    """
    # Read the AnnData object
    adata = sc.read_h5ad(adata_file_path)
    
    # Read the list of gene names
    marker_gene = pd.read_csv(genes_file_path)
    marker_gene_names = marker_gene['GeneName'].unique()

    # Filter the data based on the gene names in var_names
    filtered_adata = adata[:, adata.var_names.isin(marker_gene_names)]
    
    # Save the filtered AnnData object
    filtered_adata.write(output_file_path)
    print(f"Filtered AnnData object saved to {output_file_path}")

def filter_adata_by_genes_data(adata, genes_file_path):
    """
    Filter an AnnData object based on a list of genes and return the new data.

    Parameters:
    adata: Input AnnData object
    genes_file_path (str): File path to a CSV file containing gene names.
    """
    # Use the provided AnnData object
    adata = adata
    
    # Read the list of gene names
    marker_gene = pd.read_csv(genes_file_path)
    marker_gene_names = marker_gene['GeneName'].unique()

    # Filter the data based on the gene names in var_names
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

    # Build KDTree
    kdtree = KDTree(row_col_data)

    # Build BallTree
    balltree = BallTree(row_col_data)

    # Get the indices of the closest k-1 points and the farthest k-1 points for each row
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
    n = len(predictions)  # Number of samples
    k = len(predictions[0])  # Number of predicted and true values per sample

    squared_diff_sum = 0.0  # Sum of squared differences

    for i in range(n):
        for j in range(k):
            squared_diff = (targets[i][j] - predictions[i][j]) ** 2
            squared_diff_sum += squared_diff

    mean_squared_error = squared_diff_sum / (n * k)  # Mean squared error
    rmse = math.sqrt(mean_squared_error)  # Root mean square error

    return rmse

def reorder_and_fill_missing_vars(test_adata, real_adata):
    # Get var_names from real_adata
    real_var_names = real_adata.var_names

    # Initialize a new matrix: number of rows = test_adata observations, number of columns = real_adata variables
    new_X = np.zeros((test_adata.shape[0], len(real_var_names)))

    # Fill in data for existing variables
    for i, var in enumerate(real_var_names):
        if var in test_adata.var_names:
            new_X[:, i] = test_adata[:, var].X.toarray().flatten()

    # Update test_adata with new matrix and variable names
    test_adata = ad.AnnData(X=sp.csr_matrix(new_X), 
                            var=pd.DataFrame(index=real_var_names), 
                            obs=test_adata.obs, 
                            obsm=test_adata.obsm, 
                            obsp=test_adata.obsp, 
                            varm=test_adata.varm, 
                            varp=test_adata.varp, 
                            uns=test_adata.uns)
    
    # Normalize the sparse matrix
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
    # Create a 5-fold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Get the list of split indices
    splits = list(kf.split(np.arange(data.shape[0])))

    # Get training and testing indices for the current fold
    train_indices, test_indices = splits[current_fold]

    # Create AnnData objects for training and testing sets
    adata_train = data[train_indices].copy()
    adata_test = data[test_indices].copy()

    # Print dataset sizes
    print("测试集大小adata_test:", adata_test.shape)
    print("训练集大小adata_train:", adata_train.shape)

    return adata_train, adata_test
