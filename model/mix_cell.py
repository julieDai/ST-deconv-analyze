import os
import numpy as np
import time
import random
random.seed(42)
from scipy.sparse import csr_matrix, vstack

def generate_random_ratio(total_rows, celltype_list):
    """
    Generate a list of random ratio dictionaries for a given list of cell types.
    
    Each dictionary contains cell types as keys and their corresponding ratios as values.
    The sum of all ratios in each dictionary is equal to 1.
    
    Parameters:
    - total_rows (int): Total number of rows or data points.
    - celltype_list (list): List of cell type names.
    
    Returns:
    - ratios (list): A list of dictionaries, each containing random ratios for the data points.
    """
    ratios = []  # Initialize the list of ratios
    n = len(celltype_list)  # Number of cell types

    for i in range(total_rows):
        ratio = {}  # Ratio dictionary for each data point
        total_sum = 0  # Total sum of assigned ratios

        # # Generate random ratios for each cell type (except the last one)
        for j in range(n - 1):
            # Generate a random ratio, ensuring the total does not exceed 1
            ratio[celltype_list[j]] = random.uniform(0, 1 - total_sum)
            total_sum += ratio[celltype_list[j]]  # Update the total assigned ratio

        # Calculate the ratio for the last cell type to ensure the total equals 1
        last_ratio = 1 - total_sum
        ratio[celltype_list[n - 1]] = last_ratio

        ratios.append(ratio)  # Add the current ratio dictionary to the list
    print("generate_random_ratio finished")


    return ratios

def select_rows_by_ratio(data_list: dict, ratio, sample_num, merge_row):
    """
    Select rows from each cell type based on the given ratios.
    
    Parameters:
    - data_list (dict): A dictionary where keys are cell types and values are lists of single-cell data indices for each type.
    - ratio (list): A list of dictionaries, each containing the cell type ratios for one data point.
    - merge_rows (int): Total number of rows to be merged.
    
    Returns:
    - spot_rows (list): A list of indices representing the selected rows based on the ratios.
    """
    spot_rows = []  # Initialize the final list of selected rows


    # Iterate over each ratio dictionary, where each dictionary represents the ratio configuration for one data point
    for i in range(len(ratio)):
        selected_rows = []  # Initialize the list of selected row indices for the current data point

        # Iterate through each cell type and its corresponding data
        for cell_type, data in data_list.items():
            # Calculate the number of rows to select based on the cell type ratio and merge_rows
            num_selected = int(sample_num * ratio[i][cell_type] * merge_row)

            # Randomly select the specified number of rows from the cell type's data, allowing duplicates
            selected_rows.extend(random.choices(data, k=num_selected))

        # Add the selected row indices for the current data point to the final list
        spot_rows.append(selected_rows)
    print("select_rows_by_ratio finished")

    return spot_rows

class Mix:
    def __init__(self, single_cell_data, celltype_list, sample_num, merge_rows, filepath):
        # Initialize the Mix class to process and prepare the gene expression dataset

        # Store gene expression matrix and cell type labels    
        self.gene_expression_matrix = single_cell_data.X
        self.cell_type_matrix = np.array(single_cell_data.obs_names[:]) # h5ad
        # self.cell_type_matrix = np.array(single_cell_data.obs[:])  # For single-cell data read from CSV

        # Store other parameters: cell type list, sample count, number of merged rows, and file path
        self.cell_type_list = celltype_list
        self.sample_num = sample_num
        self.merge_rows = merge_rows
        self.filepath = os.path.join(filepath, 'one_type_to_gene_index')
        os.makedirs(self.filepath, exist_ok=True)

        # Run key functions to create cell type indices and generate random cell indices
        self.is_cell_type_index = self.get_cell_type_index()
        self.ratio = generate_random_ratio(self.merge_rows, self.cell_type_list)
        self.random_cell_index = self.get_random_cell()
        print("Mix __init__ finished")


    def get_cell_type_index(self):
        # Generate an index array for each cell type
        start_time = time.time()

        for cell_type in self.cell_type_list:
            print(f'Current For: {cell_type}')
            # Use list comprehension to find indices corresponding to the current cell type
            celltype_indices = [idx for idx, name in enumerate(self.cell_type_matrix)
                                if name.split('.')[0] == cell_type]  # h5ad

            # celltype_indices = [idx for idx, name in enumerate(self.cell_type_matrix)
            #                     if name == cell_type]  # For single-cell data read from CSV
            elapsed_time = time.time() - start_time
            print(f"Processed {cell_type}, shape:{len(celltype_indices)}. "
                  f"Elapsed time: {elapsed_time:.2f} seconds.")

            # Save the index array for each cell type to a .npy file    
            filename = f"{cell_type.replace('/', '&')}_to_gene.npy"
            np.save(os.path.join(self.filepath, filename), np.array(celltype_indices))
        print("get_cell_type_index finished")
            

        return "completed index create"

    def get_random_cell(self):
        # Select random cell row indices based on the generated ratios
        data_list = {}
        for cell_type in self.cell_type_list:
            filename = f"{cell_type.replace('/', '&')}_to_gene.npy"
            # Load previously saved indices for each cell type
            cell_array = np.load(os.path.join(self.filepath, filename))
            data_list[cell_type] = cell_array.tolist()

        # Use the given ratios and merge_rows to select random cell row indices
        random_cell_index = select_rows_by_ratio(data_list, self.ratio, self.sample_num, self.merge_rows)
        print("get_random_cell finished")

        return random_cell_index
    
    def gain_index_matrix(self):
        merge_rows = self.merge_rows
        # Preallocate a list to store the sum of each sample
        rows_list = []
        for i in range(merge_rows):
            # Directly calculate the sum of selected rows
            selected_rows_sum = np.sum(self.gene_expression_matrix[self.random_cell_index[i], :].toarray(), axis=0)
            # Convert the result to CSR format and add it to the list
            rows_list.append(csr_matrix(selected_rows_sum))

            if i % 10 == 0:
                print(f"Processed {i} merged.")

        # Stack all rows at once using vstack
        random_cell_csr = vstack(rows_list)
        print("gain_index_matrix finished")

        return random_cell_csr
