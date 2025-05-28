from ..dataset_config import Dataset
from ..options import get_option_list
from ..model.mix_cell import Mix
from ..utils.utils import merge_csr_matrices1, filter_adata_by_genes
import anndata as ad
import os
import ast



# Get option list / initialize dataset
# option = get_option_list()
# Get override options from environment variables
override_options = os.getenv('OVERRIDE_OPTIONS')
option = ast.literal_eval(override_options)  # Parse the string into a dictionary
dataset = Dataset(option)

try:
    # ST-deconv simulated data
    if dataset.bool_simu == True:
        single_cell_data = dataset.load_single_cell_data()
        real_spatial_data = dataset.load_real_spatial_data()

        file_dirc_simu_process = f'{dataset.save_results_dir}{dataset.ST_deconv_simu_process_dir}'
        # Data preprocessing and simulation
        mix_RNA_seq = Mix(single_cell_data, dataset.celltype_list, dataset.simu_sample_size, 
                        dataset.simu_sample_num, file_dirc_simu_process)

        random_cell_csr = mix_RNA_seq.gain_index_matrix()

        adata_simu, adata_real = merge_csr_matrices1(random_cell_csr, real_spatial_data, single_cell_data.var_names, 
                                        real_spatial_data.var_names, mix_RNA_seq.ratio, file_dirc_simu_process, dataset.bool_simu)
        

        if dataset.bool_ttest == True:
              filter_adata_by_genes(f'{file_dirc_simu_process}conjustion_data/adata_real.h5ad',
                                        dataset.ttest_genes_list_dir,
                                    f'{file_dirc_simu_process}conjustion_data/adata_real.h5ad')
              filter_adata_by_genes(f'{file_dirc_simu_process}conjustion_data/adata_simu.h5ad',
                                        dataset.ttest_genes_list_dir,
                                    f'{file_dirc_simu_process}conjustion_data/adata_simu.h5ad')
              


    # Use pre-existing simulated data
    else:
        simu_spatial_data = dataset.load_simu_data()
        real_spatial_data = dataset.load_real_spatial_data()
        adata_simu, adata_real = merge_csr_matrices1(simu_spatial_data, real_spatial_data, simu_spatial_data.var_names, 
                                        real_spatial_data.var_names, simu_spatial_data.obs, f'{dataset.save_results_dir}Othersimu/', dataset.bool_simu)
        if dataset.bool_ttest == True:
              filter_adata_by_genes(f'{dataset.save_results_dir}Othersimu/conjustion_data/adata_real.h5ad',
                                        dataset.ttest_genes_list_dir,
                                    f'{dataset.save_results_dir}Othersimu/conjustion_data/adata_real.h5ad')
              filter_adata_by_genes(f'{dataset.save_results_dir}Othersimu/conjustion_data/adata_simu.h5ad',
                                        dataset.ttest_genes_list_dir,
                                    f'{dataset.save_results_dir}Othersimu/conjustion_data/adata_simu.h5ad')
              

except Exception as e:
            print(f"Error occurred while selecting whether to use ST-deconv simulated data: {e}")








    
            

    








