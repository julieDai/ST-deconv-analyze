import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp
from .options import get_base_option_list
from .utils.utils import filter_adata_by_genes_data
import re


from tqdm import tqdm


class Dataset:
    def __init__(self, option_list):

        # Dataset name
        self.dataset_name = option_list.get('dataset_name')
        
        # Whether to use ST-deconv simulation method [1: ST-deconv, 0: other methods]
        self.bool_simu = option_list.get('bool_simu')
        
        # Dataset directory
        self.dataset_dir = option_list.get('dataset_dir')
        
        # Single-cell transcriptomics data filename
        self.single_cell_dataset_name = option_list.get('single_cell_dataset_name')

        # Label file name for single-cell transcriptomics data
        self.single_cell_dataset_label = option_list.get('single_cell_dataset_label')
        
        # Number of single-cell samples per spatial transcriptomics spot
        self.simu_sample_size = option_list.get('simu_sample_size')
        
        # Number of spatial transcriptomics samples
        self.simu_sample_num = option_list.get('simu_sample_num')

        # Directory for ST-deconv intermediate process files
        self.ST_deconv_simu_process_dir = option_list['ST-deconv_simu_process_dir']
        
        # Default storage location for simulated data (non-ST-deconv)
        self.simu_data_dir = option_list.get('simu_data_dir')
        self.simu_expression = option_list.get('simu_expression')
        self.simu_label = option_list.get('simu_label')
        
        # Real spatial transcriptomics data corresponding to the single-cell data
        self.real_dataset_dir = option_list.get('real_dataset_dir')

        # Whether to include spatial information
        self.bool_cl = option_list.get('bool_cl')

        # Whether to use t-test
        self.bool_ttest = option_list.get('bool_ttest')
        
        # p-value threshold for t-test
        self.p_value_ttest = option_list.get('p-vaule_ttest')
        
        # Directory for the gene list used in t-test
        self.ttest_genes_list_dir = option_list.get('ttest_genes_list_dir')

        # Number of AE-DAN alternating training epochs
        self.AE_DAN_epoch = option_list.get('AE_DAN_epoch')
        
        # AE training parameter - batch size
        self.AE_batch_size = option_list.get('AE_batch_size')
        
        # AE training parameter - number of epochs
        self.AE_epochs = option_list.get('AE_epochs')
        
        # AE training parameter - learning rate
        self.AE_learning_rate = option_list.get('AE_learning_rate')
        
        # Whether to use DAN
        self.bool_DAN = option_list.get('bool_DAN')

        # DAN training parameter - batch size
        self.DAN_batch_size = option_list.get('DAN_batch_size')
        
        # DAN training parameter - number of epochs
        self.DAN_epochs = option_list.get('DAN_epochs')
        
        # DAN training parameter - learning rate
        self.DAN_learning_rate = option_list.get('DAN_learning_rate')

        # Path to test dataset [AnnData format]
        self.test_data_dir = option_list.get('test_data_dir')
        
        # Proportion of simulated data used for test split
        self.test_dataset_split = float(option_list.get('test_dataset_split'))
        
        # Settings for external test dataset
        self.other_test_dataset_bool = option_list.get('other_test_dataset_bool')
        self.other_test_dataset_name = option_list.get('other_test_dataset_name')
        self.other_test_dataset_label_name = option_list.get('other_test_dataset_label_name')
        self.other_test_dataset_batch_size = option_list.get('other_test_dataset_batch_size')

        # Five-fold cross-validation parameters
        self.bool_fiveTest = option_list.get('bool_fiveTest')
        self.fiveTest_fold = option_list.get('fiveTest_fold')
        
        # Directory to save results
        self.save_results_dir = option_list.get('SaveResultsDir')

        # List of cell types
        self.celltype_list = self.get_celltype_list()
        
    def load_single_cell_data(self):
            # Load single-cell transcriptomics data
            try:
                if self.single_cell_dataset_name.endswith('h5ad'):
                    single_cell_data = ad.read_h5ad(f"{self.dataset_dir}{self.single_cell_dataset_name}")
                    print(f"Loaded single-cell dataset from {self.dataset_dir}{self.single_cell_dataset_name}")
                    return single_cell_data
                elif self.single_cell_dataset_name.endswith('csv'):
                    # Data
                    # Define file paths
                    file_name = f"{self.dataset_dir}{self.single_cell_dataset_name}"
                    file_label = f"{self.dataset_dir}{self.single_cell_dataset_label}"

                    # Count the number of lines for the progress bar
                    # data_df = pd.read_csv(file_name, sep='\t' if file_name.endswith('.tsv') else ',')

                    # Use tqdm to wrap the file object before opening it
                    with open(file_name, 'r') as f:
                        # First, get the total number of lines for the progress bar
                        total_lines = sum(1 for line in f)
                    
                    # Use tqdm with pd.read_csv to show the progress bar
                    # with tqdm(total=total_lines, desc="Reading CSV", unit="lines") as pbar:
                    #     data_df = pd.read_csv(file_name, sep='\t' if file_name.endswith('.tsv') else ',', 
                    #                           iterator=True, chunksize=1000)  # Read 1000 lines at a time
                    #
                    #     # Load each chunk into data_df and update the progress bar
                    #     data_df = pd.concat([chunk for chunk in data_df], ignore_index=True)
                    #     pbar.update(total_lines)
                    with tqdm(total=total_lines, desc="Reading CSV", unit="lines") as pbar:
                        # Read data in chunks
                        data_chunks = []
                        for chunk in pd.read_csv(file_name, sep='\t' if file_name.endswith('.tsv') else ',', 
                                                 iterator=True, chunksize=1000):
                            data_chunks.append(chunk)  # Store each chunk in the list
                            pbar.update(len(chunk))    # Update the progress bar for each chunk
                    
                        # Combine all chunks into a single DataFrame
                        data_df = pd.concat(data_chunks, ignore_index=True)
                    
                    # Transpose the data so that rows represent samples (cells), and columns represent variables (genes)
                    data_df = data_df.transpose()
                    data_df.columns = data_df.iloc[0]
                    data_df = data_df.drop(data_df.index[0])

                    # Read the cell type label file; the file should contain only one column
                    label_df = pd.read_csv(file_label, sep='\t' if file_label.endswith('.tsv') else ',')
                    label_df.columns = ['cellType']
                    
                    # Create the obs DataFrame, with indices aligned to data_df; use one or more columns as fields in obs
                    obs_df = pd.DataFrame(index=data_df.index, data=label_df.values, columns=label_df.columns)
                    
                    # Convert data_df to a sparse matrix format
                    sparse_matrix = sp.csr_matrix(np.array(data_df, dtype=np.float64))
                    
                    # Create the var DataFrame containing the original column names of data_df
                    var_df = pd.DataFrame(index=data_df.columns)
                    
                    # Create the AnnData object
                    adata = ad.AnnData(X=sparse_matrix, var=var_df, obs=obs_df)
                    
                    # Save the AnnData object to a file
                    adata.write(f"{self.dataset_dir}single_cell/count.h5ad")
                    
                    return adata


            except Exception as e:
                print(f"Error loading single-cell dataset: {e}")
                return None    
            
    def load_real_spatial_data(self):
        # Load real spatial transcriptomics data and convert it to AnnData format
        try:
            if self.real_dataset_dir.endswith('.csv') or self.real_dataset_dir.endswith('.tsv'):
                real_data_df = pd.read_csv(self.real_dataset_dir, sep='\t' if self.real_dataset_dir.endswith('.tsv') else ',')
                real_data_df.index = real_data_df.iloc[:, 0]
                real_data_df = real_data_df.drop(real_data_df.columns[0], axis=1)
                
                # Select only numeric columns
                real_data_df_numeric = real_data_df.select_dtypes(include=[np.number])
                
                # Convert the DataFrame to a sparse matrix
                real_data_csr = sp.csr_matrix(real_data_df_numeric.values)
                
                # Create an AnnData object and set X as the sparse matrix
                var_df = pd.DataFrame(index=real_data_df_numeric.columns)
                obs_df = pd.DataFrame(index=real_data_df_numeric.index)
                real_data = ad.AnnData(X=real_data_csr, var=var_df, obs=obs_df)
                
                print(f"Loaded real spatial transcriptomics data from {self.real_dataset_dir}")
            else:
                if self.real_dataset_dir.endswith('.h5ad'):
                    real_data = ad.read_h5ad(self.real_dataset_dir)
                    
                    # Ensure X is a sparse matrix
                    if not sp.issparse(real_data.X):
                        real_data.X = sp.csr_matrix(real_data.X)
                    
                    print(f"Loaded real spatial transcriptomics data from {self.real_dataset_dir}")
                
            return real_data
        except Exception as e:
            print(f"Error loading real spatial transcriptomics data: {e}")
            return None
        
    def load_test_data(self):
        # Load simulated test data
        file_name = f"{self.test_data_dir}{self.other_test_dataset_name}"
        file_label = f"{self.test_data_dir}{self.other_test_dataset_label_name}"
    
        try:
            if file_name.endswith('.csv') or file_name.endswith('.tsv'):
                test_data_df = pd.read_csv(file_name, sep='\t' if file_name.endswith('.tsv') else ',')
    
                # Transpose data
                test_data_df = test_data_df.transpose()
                test_data_df.columns = test_data_df.iloc[0]
                test_data_df = test_data_df.drop(test_data_df.index[0])
                
                # Remove the first column and select only numeric columns
                test_data_df_numeric = test_data_df.values
                test_data_df_numeric = np.array(test_data_df, dtype=np.float64)
                
                # Convert DataFrame to sparse matrix
                test_data_csr = sp.csr_matrix(test_data_df_numeric)
                
                # Create AnnData object and ensure X is a sparse matrix
                var_df = pd.DataFrame(index=test_data_df.columns)
                # Add label as obs in AnnData object
                if file_label.endswith('.csv') or file_label.endswith('.tsv'):
                    test_label_df = pd.read_csv(file_label, sep='\t' if file_label.endswith('.tsv') else ',')
                obs_df = pd.DataFrame(index=test_data_df.index, data=test_label_df.values, columns=test_label_df.columns)
                test_data = ad.AnnData(X=test_data_csr, var=var_df, obs=obs_df)
    
                # Parse obs_names to extract x and y coordinates
                x_values = []
                y_values = []
    
                for name in test_data.obs_names:
                    match = re.match(r'(\d+\.\d+)x(\d+\.\d+)', name)
                    if match:
                        x, y = match.groups()
                        x_values.append(float(x))
                        y_values.append(float(y))
    
                # Store x and y in new columns in obs
                test_data.obs['x'] = x_values
                test_data.obs['y'] = y_values
                    
            if self.test_data_dir.endswith('.h5ad'):
                test_data = ad.read_h5ad(self.test_data_dir)
    
            # Apply t-test filtering if enabled
            if self.bool_ttest == True:
                test_data = filter_adata_by_genes_data(test_data, self.ttest_genes_list_dir)
    
            print(f"Loaded test dataset from {self.test_data_dir}")
            return test_data
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            return None

    def load_simu_data(self):
        # Load simulated data not generated by ST-deconv
        file_name = f'{self.simu_data_dir}{self.simu_expression}'
        file_label = f'{self.simu_data_dir}{self.simu_label}'
    
        try:
            if file_name.endswith('.csv') or file_name.endswith('.tsv'):
                simu_data_df = pd.read_csv(file_name, sep='\t' if file_name.endswith('.tsv') else ',')
    
                # Transpose the data
                simu_data_df = simu_data_df.transpose()
                simu_data_df.columns = simu_data_df.iloc[0]
                simu_data_df = simu_data_df.drop(simu_data_df.index[0])
                
                # Remove the first column and retain only numeric values
                simu_data_df_numeric = simu_data_df.values
                simu_data_df_numeric = np.array(simu_data_df, dtype=np.float64)
                
                # Convert DataFrame to sparse matrix
                simu_data_csr = sp.csr_matrix(simu_data_df_numeric)
                
                # Create AnnData object and set X to sparse matrix
                var_df = pd.DataFrame(index=simu_data_df.columns)
                
                # Add label as obs into AnnData object
                if file_label.endswith('.csv') or file_label.endswith('.tsv'):
                    simu_label_df = pd.read_csv(file_label, sep='\t' if file_label.endswith('.tsv') else ',')
                obs_df = pd.DataFrame(index=simu_data_df.index, data=simu_label_df.values, columns=simu_label_df.columns)
                simu_data = ad.AnnData(X=simu_data_csr, var=var_df, obs=obs_df)
    
            if self.simu_data_dir.endswith('.h5ad'):
                simu_data = ad.read_h5ad(self.simu_data_dir)
    
            print(f"Loaded simulated dataset from {self.simu_data_dir}")
            return simu_data
        except Exception as e:
            print(f"Error loading simulated dataset: {e}")
            return None

    def load_ttest_genes(self):
        # Load t-test gene name list
        try:
            ttest_genes = pd.read_csv(self.ttest_genes_list_dir, sep='\t', header=None).iloc[:, 0].tolist()
            print(f"Loaded ttest genes list from {self.ttest_genes_list_dir}")
            return ttest_genes
        except Exception as e:
            print(f"Error loading ttest genes list: {e}")
            return None
        
    def get_celltype_list(self):
        single_cell_data = self.load_single_cell_data()
        # Get the number of rows in the X matrix
        cell_type_matrix = np.array(single_cell_data.obs_names[:])
        # cell_type_matrix = np.array(single_cell_data.obs['cellType'])
        num_cells = cell_type_matrix.shape[0]

        # Create a list to store unique cell type names
        celltype_list = []

        # Define the list of cell indices to iterate over
        cell_indices = range(num_cells)

        for cell_idx in cell_indices:
            celltype_name = cell_type_matrix[cell_idx].split('.')[0]
            if celltype_name not in celltype_list:
                celltype_list.append(celltype_name)

        return celltype_list
    
# # Test reading data
# option = get_base_option_list()  # Parse options from dictionary
# dataset = Dataset(option)
# dataset.load_single_cell_data()


