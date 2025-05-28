from collections import defaultdict

option_list = defaultdict(list)
updated_option_lists = defaultdict(list)

def get_base_option_list():

	dataset = 'MOB_CARD_simu'

	if  dataset == 'MOB_CARD_simu':
		option_list['dataset_name'] = 'MOB_CARD_simu'

		# ---------------------Data preprocessing and training/testing dataset------------------------

	        # Data simulation
	        option_list['bool_simu'] = 1  # Boolean value, [1: ST-deconv, 0: other simulation methods]
	
	        # Data simulation preprocessing
	        option_list['dataset_dir'] = 'ST-deconv/data/MOB/trainset/'
	        option_list['single_cell_dataset_name'] = 'single_cell/count.h5ad'  # Single-cell transcriptomic data
	        option_list['single_cell_dataset_label'] = 'single_cell/cell_types_specific.csv'  # Cell type labels
	        option_list['simu_sample_size'] = 10  # Number of single-cell samples per spatial transcriptomic spot
	        option_list['simu_sample_num'] = 282  # Number of samples in the simulated spatial transcriptomic data
	        option_list['ST-deconv_simu_process_dir'] = 'ST-deconv_simu_temp/'
	        option_list['simu_data_dir'] = 'ST-deconv/data/MOB/trainset/Card_simu_data/31/'  # Simulated data path (non-ST-deconv)
	        option_list['simu_expression'] = 'pseudo_data_31.csv'
	        option_list['simu_label'] = 'pseudo_data_ratio_31.csv'

	        # DAN preprocessing (real spatial transcriptomic data corresponding to simulated data)
	        option_list['real_dataset_dir'] = 'ST-deconv/data/MOB/trainset/real_spatial_12.tsv'
	
	        # Incorporate spatial information: CL
	        option_list['bool_cl'] = 1
	
	        # t-test parameters
	        option_list['bool_ttest'] = 1
	        option_list['p-vaule_ttest'] = ''
	        option_list['ttest_genes_list_dir'] = 'ST-deconv/data/MOB/trainset/ttest/significant_genes_ttest_0.01.csv'

        # ----------------------- Training parameters------------------------------------------
		# Total training epochs
		option_list['AE_DAN_epoch'] = 60

		# AE training parameters
		option_list['AE_batch_size'] = 32
		option_list['AE_epochs'] =  30
		option_list['AE_learning_rate'] = 0.0003

		# DAN training parameters
		option_list['bool_DAN'] = 1
		option_list['DAN_batch_size'] = 32
		option_list['DAN_epochs'] =  30
		option_list['DAN_learning_rate'] = 0.0003

		#-------------------------Testing parameters------------------------------
	        # Default testing data path
	        option_list['test_data_dir'] = 'ST-deconv/data/MOB/testset/'  # Path to AnnData formatted test data
	        option_list['test_dataset_split'] = 0.1  # Proportion of simulated data used for testing
	
	        # Whether to use other external test datasets and batch size
	        option_list['other_test_dataset_bool'] = '0'  # Boolean value for external test datasets
	        option_list['other_test_dataset_name'] = 'Card_simu_data/02/pseudo_data_02.csv'
	        option_list['other_test_dataset_label_name'] = 'Card_simu_data/02/pseudo_data_ratio_02.csv'
	        option_list['other_test_dataset_batch_size'] = 282
	
	        # 5-fold cross-validation parameters
	        option_list['bool_fiveTest'] = 1  # Whether to use 5-fold cross-validation
	        option_list['fiveTest_fold'] = 0  # Index of the test fold
	
	        # Output parameter
	        option_list['SaveResultsDir'] = f'/data/ST-deconv/experiment/CL4.0/'
	return option_list

overrides = [
	('AE', {'bool_simu': 0, 'bool_cl': 0, 'bool_ttest': 0, 'bool_DAN': 0}),
	('AE_simu', {'bool_simu': 1, 'bool_cl': 0, 'bool_ttest': 0, 'bool_DAN': 0}),
	('AE_spatial', {'bool_simu': 0, 'bool_cl': 1, 'bool_ttest': 0, 'bool_DAN': 0}),
	('AE_ttest', {'bool_simu': 0, 'bool_cl': 0, 'bool_ttest': 1, 'bool_DAN': 0}),
	('AE_DAN', {'bool_simu': 0, 'bool_cl': 0, 'bool_ttest': 0, 'bool_DAN': 1}),
	('AE_simu_spatial', {'bool_simu': 1, 'bool_cl': 1, 'bool_ttest': 0, 'bool_DAN': 0}),
	('AE_simu_ttest', {'bool_simu': 1, 'bool_cl': 0, 'bool_ttest': 1, 'bool_DAN': 0}),
	('AE_simu_DAN', {'bool_simu': 1, 'bool_cl': 0, 'bool_ttest': 0, 'bool_DAN': 1}),
	('AE_spatial_ttest', {'bool_simu': 0, 'bool_cl': 1, 'bool_ttest': 1, 'bool_DAN': 0}),
	('AE_spatial_DAN', {'bool_simu': 0, 'bool_cl': 1, 'bool_ttest': 0, 'bool_DAN': 1}),
	('AE_ttest_DAN', {'bool_simu': 0, 'bool_cl': 0, 'bool_ttest': 1, 'bool_DAN': 1}),
	('AE_simu_spatial_ttest', {'bool_simu': 1, 'bool_cl': 1, 'bool_ttest': 1, 'bool_DAN': 0}),
	('AE_simu_spatial_DAN', {'bool_simu': 1, 'bool_cl': 1, 'bool_ttest': 0, 'bool_DAN': 1}),
	('AE_simu_ttest_DAN', {'bool_simu': 1, 'bool_cl': 0, 'bool_ttest': 1, 'bool_DAN': 1}),
	('AE_spatial_ttest_DAN', {'bool_simu': 0, 'bool_cl': 1, 'bool_ttest': 1, 'bool_DAN': 1}),
	('AE_simu_spatial_ttest_DAN', {'bool_simu': 1, 'bool_cl': 1, 'bool_ttest': 1, 'bool_DAN': 1}),

	
    ('01', {'simu_data_dir':'ST-deconv/data/MOB/trainset/Card_simu_data/01/', 
            'simu_expression':'pseudo_data_01.csv', 
            'simu_label':'pseudo_data_ratio_01.csv', 
            'other_test_dataset_name':'Card_simu_data/31/pseudo_data_31.csv', 
            'other_test_dataset_label_name' : 'Card_simu_data/31/pseudo_data_ratio_31.csv'}),
    
    ('02', {'simu_data_dir':'ST-deconv/data/MOB/trainset/Card_simu_data/02/', 
            'simu_expression':'pseudo_data_02.csv', 
            'simu_label':'pseudo_data_ratio_02.csv', 
            'other_test_dataset_name':'Card_simu_data/32/pseudo_data_32.csv', 
            'other_test_dataset_label_name' : 'Card_simu_data/32/pseudo_data_ratio_32.csv'}),
    
    ('03', {'simu_data_dir':'ST-deconv/data/MOB/trainset/Card_simu_data/03/', 
            'simu_expression':'pseudo_data_03.csv', 
            'simu_label':'pseudo_data_ratio_03.csv', 
            'other_test_dataset_name':'Card_simu_data/33/pseudo_data_33.csv', 
            'other_test_dataset_label_name' : 'Card_simu_data/33/pseudo_data_ratio_33.csv'}),
    
    ('04', {'simu_data_dir':'ST-deconv/data/MOB/trainset/Card_simu_data/04/', 
            'simu_expression':'pseudo_data_04.csv', 
            'simu_label':'pseudo_data_ratio_04.csv', 
            'other_test_dataset_name':'Card_simu_data/34/pseudo_data_34.csv', 
            'other_test_dataset_label_name' : 'Card_simu_data/34/pseudo_data_ratio_34.csv'}),
    
    ('05', {'simu_data_dir':'ST-deconv/data/MOB/trainset/Card_simu_data/05/', 
            'simu_expression':'pseudo_data_05.csv', 
            'simu_label':'pseudo_data_ratio_05.csv', 
            'other_test_dataset_name':'Card_simu_data/35/pseudo_data_35.csv', 
            'other_test_dataset_label_name' : 'Card_simu_data/35/pseudo_data_ratio_35.csv'}),
    
    ('31', {'simu_data_dir':'ST-deconv/data/MOB/trainset/Card_simu_data/31/', 
            'simu_expression':'pseudo_data_31.csv', 
            'simu_label':'pseudo_data_ratio_31.csv', 
            'other_test_dataset_name':'Card_simu_data/01/pseudo_data_01.csv', 
            'other_test_dataset_label_name' : 'Card_simu_data/01/pseudo_data_ratio_01.csv'}),
    
    ('32', {'simu_data_dir':'ST-deconv/data/MOB/trainset/Card_simu_data/32/', 
            'simu_expression':'pseudo_data_32.csv', 
            'simu_label':'pseudo_data_ratio_32.csv', 
            'other_test_dataset_name':'Card_simu_data/02/pseudo_data_02.csv', 
            'other_test_dataset_label_name' : 'Card_simu_data/02/pseudo_data_ratio_02.csv'}),
    
    ('33', {'simu_data_dir':'ST-deconv/data/MOB/trainset/Card_simu_data/33/', 
            'simu_expression':'pseudo_data_33.csv', 
            'simu_label':'pseudo_data_ratio_33.csv', 
            'other_test_dataset_name':'Card_simu_data/03/pseudo_data_03.csv', 
            'other_test_dataset_label_name' : 'Card_simu_data/03/pseudo_data_ratio_03.csv'}),
    
    ('34', {'simu_data_dir':'ST-deconv/data/MOB/trainset/Card_simu_data/34/', 
            'simu_expression':'pseudo_data_34.csv', 
            'simu_label':'pseudo_data_ratio_34.csv', 
            'other_test_dataset_name':'Card_simu_data/04/pseudo_data_04.csv', 
            'other_test_dataset_label_name' : 'Card_simu_data/04/pseudo_data_ratio_04.csv'}),
    
    ('35', {'simu_data_dir':'ST-deconv/data/MOB/trainset/Card_simu_data/35/', 
            'simu_expression':'pseudo_data_35.csv', 
            'simu_label':'pseudo_data_ratio_35.csv', 
            'other_test_dataset_name':'Card_simu_data/05/pseudo_data_05.csv', 
            'other_test_dataset_label_name' : 'Card_simu_data/05/pseudo_data_ratio_05.csv'}),
]

overrides_train_dataset = []

override = None
overide_trainset = None
def set_override_value(key, key_trainset):
    global override, overrides_train_dataset  # Define two global variables
    for item in overrides:
        if item[0] == key:
            override = item
        elif item[0] == key_trainset:
            overrides_train_dataset = item
            break  # Only terminate the loop when key_trainset is found


def get_option_list(result_name):
	option_list = get_base_option_list()
	local_option_list = option_list.copy()

	if override:
		for key, value in override[1].items():
			local_option_list[key] = value
		experiment_dir = local_option_list['SaveResultsDir']
		experiment_name = override[0]
		local_option_list['SaveResultsDir'] = f'{experiment_dir}{result_name}/{experiment_name}/'
	if overrides_train_dataset:
		for key, value in overrides_train_dataset[1].items():
			local_option_list[key] = value
	updated_option_lists = local_option_list
	return updated_option_lists


