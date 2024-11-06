from collections import defaultdict

option_list = defaultdict(list)
updated_option_lists = defaultdict(list)

def get_base_option_list():

	dataset = 'MOB_CARD_simu'

	if  dataset == 'MOB_CARD_simu':
		option_list['dataset_name'] = 'MOB_CARD_simu'

		# ---------------------数据预处理和训练测试数据集------------------------

		# 数据模拟
		option_list['bool_simu'] = 1 # bool值，[1：ST-deconv,0:其他数据模拟方法]

		# 数据模拟预处理
		option_list['dataset_dir']='ST-deconv/data/MOB/trainset/'
		option_list['single_cell_dataset_name'] = 'single_cell/count.h5ad'		#单细胞转录组数据
		option_list['single_cell_dataset_label'] = 'single_cell/cell_types_specific.csv' #单细胞转录组数据细胞类型标签
		option_list['simu_sample_size'] = 10		# 每个空间转录组spot的单细胞数据量
		option_list['simu_sample_num'] = 282	#与空间转录组数据的样本量*10
		option_list['ST-deconv_simu_process_dir'] = 'ST-deconv_simu_temp/'
		option_list['simu_data_dir'] = 'ST-deconv/data/MOB/trainset/Card_simu_data/31/'       # 外来的模拟数据的存储位置(生成的模拟数据的存储位置：'ST-deconv/data/trainset/ST-deconv_simu_temp/conjustion_data/')
		option_list['simu_expression'] = 'pseudo_data_31.csv'
		option_list['simu_label'] = 'pseudo_data_ratio_31.csv'

        # DAN的数据预处理（模拟数据对应的真实空间转录组数据）
		option_list['real_dataset_dir'] = 'ST-deconv/data/MOB/trainset/real_spatial_12.tsv'	#与单细胞转录组数据对应的空间转录组数据
		
		# 引入空间信息:CL
		option_list['bool_cl'] = 1

        # ttest的参数
		option_list['bool_ttest'] = 1
		option_list['p-vaule_ttest'] = ''
		option_list['ttest_genes_list_dir'] = 'ST-deconv/data/MOB/trainset/ttest/significant_genes_ttest_0.01.csv'
		
        # -----------------------训练参数------------------------------------------
		# 训练的轮次
		option_list['AE_DAN_epoch'] = 60

		# AE的训练参数
		option_list['AE_batch_size'] = 32
		option_list['AE_epochs'] =  30
		option_list['AE_learning_rate'] = 0.0003

		# DAN的训练参数
		option_list['bool_DAN'] = 1
		option_list['DAN_batch_size'] = 32
		option_list['DAN_epochs'] =  30
		option_list['DAN_learning_rate'] = 0.0003

		#-------------------------测试参数------------------------------
		# 默认的测试数据位置
		option_list['test_data_dir'] = 'ST-deconv/data/MOB/testset/'        # anndata格式的测试数据存放的位置(当文件夹写入本变量且后缀为.h5ad触法)
		option_list['test_dataset_split'] = 0.1      # 分出的测试模拟数据的比例

		# 是否使用其他来源的测试数据以及它的batch_size
		option_list['other_test_dataset_bool'] = '0'       # (当不使用外来测试数据时为0)外来测试数据的bool值:例如只是用simu模拟（无其他来源模拟空转数据，使用st-simu空转数据时）
		option_list['other_test_dataset_name'] = 'Card_simu_data/02/pseudo_data_02.csv'       # 外来测试数据
		option_list['other_test_dataset_label_name'] = 'Card_simu_data/02/pseudo_data_ratio_02.csv'       # 外来测试数据
		option_list['other_test_dataset_batch_size'] = 282

		# 五则交叉验证参数
		option_list['bool_fiveTest'] = 1 # 是否使用五折交叉验证
		option_list['fiveTest_fold'] = 0 # 五折交叉验证此时的测试集

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
    global override, overrides_train_dataset  # 定义两个全局变量
    for item in overrides:
        if item[0] == key:
            override = item
        elif item[0] == key_trainset:
            overrides_train_dataset = item
            break  # 只有找到 key_trainset 时才终止循环


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


