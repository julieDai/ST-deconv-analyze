import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def generate_ref_matrix_location_info(adata_simu, filefold):
    def plot_cluster_scatter(data_pca, data_pca_centers, filefold1):
        plt.figure(figsize=(8, 6))
        plt.scatter(data_pca.values[:, 0], data_pca.values[:, 1], s=3, c=data_pca.values[:, 2], cmap='Accent')
        plt.scatter(data_pca_centers.values[:, 0], data_pca_centers.values[:, 1], marker='o', s=55, c='#8E00FF')
        plt.show()
        data_pca.to_csv(f'{filefold1}/data_pca.csv', index=False)

    simu_sparse = adata_simu.X
    encoded_matrix_ref_df = pd.DataFrame.sparse.from_spmatrix(simu_sparse)
    gene_index = adata_simu.var_names
    encoded_matrix_ref_df.columns = gene_index

    scaler = StandardScaler(with_mean=False)
    normalized_values = scaler.fit_transform(encoded_matrix_ref_df.values)
    df_normalized_data = pd.DataFrame(normalized_values, columns=encoded_matrix_ref_df.columns)

    kms = KMeans(n_clusters=5, init='k-means++')
    data_fig = kms.fit(df_normalized_data)
    centers = kms.cluster_centers_
    labs = kms.labels_
    df_labels = pd.DataFrame(kms.labels_)

    df_A_0 = df_normalized_data[labs == 0]
    df_A_1 = df_normalized_data[labs == 1]
    df_A_2 = df_normalized_data[labs == 2]
    df_A_3 = df_normalized_data[labs == 3]
    df_A_4 = df_normalized_data[labs == 4]
    m = np.shape(df_A_0)[1]
    df_A_0.insert(df_A_0.shape[1], 'label', 0)
    df_A_1.insert(df_A_1.shape[1], 'label', 1)
    df_A_2.insert(df_A_2.shape[1], 'label', 2)
    df_A_3.insert(df_A_3.shape[1], 'label', 3)
    df_A_4.insert(df_A_4.shape[1], 'label', 4)
    df_labels_data = pd.concat([df_A_0, df_A_1, df_A_2, df_A_3, df_A_4])

    df_centers = pd.DataFrame(centers)

    pca = PCA(n_components=2)
    pca.fit(df_normalized_data)
    data_pca = pca.transform(df_normalized_data)
    data_pca = pd.DataFrame(data_pca, columns=['x', 'y'])
    data_pca.insert(data_pca.shape[1], 'labels', labs)

    pca = PCA(n_components=2)
    pca.fit(centers)
    data_pca_centers = pca.transform(centers)
    data_pca_centers = pd.DataFrame(data_pca_centers, columns=['x', 'y'])

    plot_cluster_scatter(data_pca, data_pca_centers, filefold)

    data_pca.to_csv(f'{filefold}/data_pca.csv', index=False)
    data_pca = pd.read_csv(f'{filefold}/data_pca.csv')

    x_min, x_max = data_pca['x'].min(), data_pca['x'].max()
    y_min, y_max = data_pca['y'].min(), data_pca['y'].max()
    x_scale = (x_max - x_min) / (np.ceil(x_max) - np.floor(x_min))
    y_scale = (y_max - y_min) / (np.ceil(y_max) - np.floor(y_min))

    rounded_data_pca = data_pca.copy()
    rounded_data_pca['x'] = np.round((rounded_data_pca['x'] - x_min) / x_scale + np.floor(x_min)).astype(int)
    rounded_data_pca['y'] = np.round((rounded_data_pca['y'] - y_min) / y_scale + np.floor(y_min)).astype(int)
    x_offset = 1 - np.floor(x_min)
    y_offset = 1 - np.floor(y_min)

    rounded_data_pca = data_pca.copy()
    rounded_data_pca['x'] = np.round(rounded_data_pca['x'] - x_min + x_offset).astype(int)
    rounded_data_pca['y'] = np.round(rounded_data_pca['y'] - y_min + y_offset).astype(int)
    rounded_data_pca.to_csv(
        f'{filefold}/data_pca_neighbor.csv',
        index=False)
    adata_simu.obs= rounded_data_pca
    return adata_simu

