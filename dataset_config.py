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

        # 数据集名称
        self.dataset_name = option_list.get('dataset_name')
        
        # 是否使用ST-deconv模拟方法，bool值，[1：ST-deconv, 0:其他数据模拟方法]
        self.bool_simu = option_list.get('bool_simu')
        
        # 数据集目录
        self.dataset_dir = option_list.get('dataset_dir')
        
        # 单细胞转录组数据文件名
        self.single_cell_dataset_name = option_list.get('single_cell_dataset_name')

        # 单细胞转录组数据标签文件名
        self.single_cell_dataset_label = option_list.get('single_cell_dataset_label')
        
        # 每个空间转录组spot的单细胞数据量
        self.simu_sample_size = option_list.get('simu_sample_size')
        
        # 与空间转录组数据的样本量相同
        self.simu_sample_num = option_list.get('simu_sample_num')

        # ST-deconv处理数据时的过程数据存储地址
        self.ST_deconv_simu_process_dir = option_list['ST-deconv_simu_process_dir']
        
        # 默认的模拟数据的存储位置(使用非ST-deconv生成的模拟数据)
        self.simu_data_dir = option_list.get('simu_data_dir')
        self.simu_expression = option_list.get('simu_expression')
        self.simu_label = option_list.get('simu_label')
        
        # 与单细胞转录组数据对应的真实空间转录组数据
        self.real_dataset_dir = option_list.get('real_dataset_dir')

        # 引入空间信息
        self.bool_cl = option_list.get('bool_cl')

        # 是否使用ttest
        self.bool_ttest = option_list.get('bool_ttest')
        
        # ttest的p值参数
        self.p_value_ttest = option_list.get('p-vaule_ttest')
        
        # ttest基因列表目录
        self.ttest_genes_list_dir = option_list.get('ttest_genes_list_dir')

        # 轮次
        self.AE_DAN_epoch = option_list.get('AE_DAN_epoch')
        
        # AE的训练参数 - 批量大小
        self.AE_batch_size = option_list.get('AE_batch_size')
        
        # AE的训练参数 - 训练轮数
        self.AE_epochs = option_list.get('AE_epochs')
        
        # AE的训练参数 - 学习率
        self.AE_learning_rate = option_list.get('AE_learning_rate')
        
        # 是否使用DAN
        self.bool_DAN = option_list.get('bool_DAN')

        # DAN的训练参数 - 批量大小
        self.DAN_batch_size = option_list.get('DAN_batch_size')
        
        # DAN的训练参数 - 训练轮数
        self.DAN_epochs = option_list.get('DAN_epochs')
        
        # DAN的训练参数 - 学习率
        self.DAN_learning_rate = option_list.get('DAN_learning_rate')

        # 测试数据存储位置【anndata格式的测试数据】
        self.test_data_dir = option_list.get('test_data_dir')
        
        # simu数据分割模拟数据的比例
        self.test_dataset_split =  float(option_list.get('test_dataset_split'))
        
        # 外来测试数据的名称
        self.other_test_dataset_bool = option_list.get('other_test_dataset_bool')
        self.other_test_dataset_name = option_list.get('other_test_dataset_name')
        self.other_test_dataset_label_name = option_list.get('other_test_dataset_label_name')
        self.other_test_dataset_batch_size = option_list.get('other_test_dataset_batch_size')

        #五折验证参数
        self.bool_fiveTest = option_list.get('bool_fiveTest')
        self.fiveTest_fold = option_list.get('fiveTest_fold')
        
        # 结果保存目录
        self.save_results_dir = option_list.get('SaveResultsDir')

        #cell_type
        self.celltype_list = self.get_celltype_list()

    def load_single_cell_data(self):
            # 加载单细胞转录组数据
            try:
                if self.single_cell_dataset_name.endswith('h5ad'):
                    single_cell_data = ad.read_h5ad(f"{self.dataset_dir}{self.single_cell_dataset_name}")
                    print(f"Loaded single-cell dataset from {self.dataset_dir}{self.single_cell_dataset_name}")
                    return single_cell_data
                elif self.single_cell_dataset_name.endswith('csv'):
                    # 数据
                    # 定义文件路径
                    file_name = f"{self.dataset_dir}{self.single_cell_dataset_name}"
                    file_label = f"{self.dataset_dir}{self.single_cell_dataset_label}"

                    # 读取基因表达数据，根据文件后缀确定分隔符
                    # data_df = pd.read_csv(file_name, sep='\t' if file_name.endswith('.tsv') else ',')

                    # 打开文件前使用 tqdm 包装文件对象
                    with open(file_name, 'r') as f:
                        # 先获取文件的总行数，用于进度条
                        total_lines = sum(1 for line in f)
                        
                    # 使用 tqdm 结合 pd.read_csv 显示进度条
                    # with tqdm(total=total_lines, desc="Reading CSV", unit="lines") as pbar:
                    #     data_df = pd.read_csv(file_name, sep='\t' if file_name.endswith('.tsv') else ',', 
                    #                         iterator=True, chunksize=1000)  # 每次读取 1000 行
                        
                    #     # 逐个chunk加载到data_df，并更新进度条
                    #     data_df = pd.concat([chunk for chunk in data_df], ignore_index=True)
                    #     pbar.update(total_lines)
                    with tqdm(total=total_lines, desc="Reading CSV", unit="lines") as pbar:
                        # 按块读取数据
                        data_chunks = []
                        for chunk in pd.read_csv(file_name, sep='\t' if file_name.endswith('.tsv') else ',', 
                                                iterator=True, chunksize=1000):
                            data_chunks.append(chunk)  # 将每个块存储到列表中
                            pbar.update(len(chunk))  # 每次更新进度条

                        # 将所有块合并为一个 DataFrame
                        data_df = pd.concat(data_chunks, ignore_index=True)


                    # 转置数据，以便行表示样本（细胞），列表示变量（基因）
                    data_df = data_df.transpose()
                    data_df.columns = data_df.iloc[0]
                    data_df = data_df.drop(data_df.index[0])

                    # 读取细胞类型标签文件，标签文件中应只包含一个列
                    label_df = pd.read_csv(file_label, sep='\t' if file_label.endswith('.tsv') else ',')
                    label_df.columns = ['cellType']

                    # 创建 obs DataFrame，索引与 data_df 保持一致，单列或多列数据作为 obs 的多个字段
                    obs_df = pd.DataFrame(index=data_df.index, data=label_df.values, columns=label_df.columns)

                    # 转换 data_df 为稀疏矩阵格式
                    sparse_matrix = sp.csr_matrix(np.array(data_df, dtype=np.float64))

                    # 创建 var DataFrame，包含原始 data_df 的列名
                    var_df = pd.DataFrame(index=data_df.columns)

                    # 创建 anndata 对象
                    adata = ad.AnnData(X=sparse_matrix, var=var_df, obs=obs_df)

                    # 保存 AnnData 对象到文件
                    adata.write(f"{self.dataset_dir}single_cell/count.h5ad")

                    return adata


            except Exception as e:
                print(f"Error loading single-cell dataset: {e}")
                return None    
            
    def load_real_spatial_data(self):
        # 加载真实空间转录组数据并转换为anndata格式
        try:
            if self.real_dataset_dir.endswith('.csv') or self.real_dataset_dir.endswith('.tsv'):
                real_data_df = pd.read_csv(self.real_dataset_dir, sep='\t' if self.real_dataset_dir.endswith('.tsv') else ',')
                real_data_df.index = real_data_df.iloc[:, 0]
                real_data_df = real_data_df.drop(real_data_df.columns[0], axis=1)
                # 仅选择数值列
                real_data_df_numeric = real_data_df.select_dtypes(include=[np.number])
                
                # 将 DataFrame 转换为稀疏矩阵
                real_data_csr = sp.csr_matrix(real_data_df_numeric.values)
                
                # 创建 AnnData 对象并设置 X 为稀疏矩阵
                var_df = pd.DataFrame(index=real_data_df_numeric.columns)
                obs_df = pd.DataFrame(index=real_data_df_numeric.index)
                real_data = ad.AnnData(X=real_data_csr, var=var_df, obs =obs_df)
                
                print(f"Loaded real spatial transcriptomics data from {self.real_dataset_dir}")
            else:
                if self.real_dataset_dir.endswith('.h5ad'):
                    real_data = ad.read_h5ad(self.real_dataset_dir)
                    
                    # 确保 X 为稀疏矩阵
                    if not sp.issparse(real_data.X):
                        real_data.X = sp.csr_matrix(real_data.X)
                    
                    print(f"Loaded real spatial transcriptomics data from {self.real_dataset_dir}")
                
            return real_data
        except Exception as e:
            print(f"Error loading real spatial transcriptomics data: {e}")
            return None
        
    def load_test_data(self):
        # 加载模拟数据
        file_name = f"{self.test_data_dir}{self.other_test_dataset_name}"
        # file_name = 'ST-deconv/data/MOB/testset/Card_simu_data/01/pseudo_data_01.csv'
        file_label = f"{self.test_data_dir}{self.other_test_dataset_label_name}"
        # file_label = 'ST-deconv/data/MOB/testset/Card_simu_data/01/pseudo_data_ratio_01.csv'

        try:
            if file_name.endswith('.csv') or file_name.endswith('.tsv'):
                test_data_df = pd.read_csv(file_name, sep='\t' if file_name.endswith('.tsv') else ',')

                # 数据转置
                test_data_df = test_data_df.transpose()
                test_data_df.columns = test_data_df.iloc[0]
                test_data_df = test_data_df.drop(test_data_df.index[0])
                
                # 删除第一列并仅选择数值列
                test_data_df_numeric = test_data_df.values
                test_data_df_numeric = np.array(test_data_df, dtype=np.float64)
                
                # 将 DataFrame 转换为稀疏矩阵
                test_data_csr = sp.csr_matrix(test_data_df_numeric)
                
                # 创建 AnnData 对象并设置 X 为稀疏矩阵
                var_df = pd.DataFrame(index=test_data_df.columns)
                # 将label作为obs加入anndata对象
                if file_label.endswith('.csv') or file_label.endswith('.tsv'):
                    test_label_df = pd.read_csv(file_label, sep='\t' if file_label.endswith('.tsv') else ',')
                    # test_label_df = test_label_df.set_index(test_label_df.columns[0])
                obs_df = pd.DataFrame(index=test_data_df.index, data=test_label_df.values, columns= test_label_df.columns)
                test_data = ad.AnnData(X=test_data_csr, var=var_df, obs = obs_df)

                # 解析 obs_names 并提取 x 和 y
                x_values = []
                y_values = []

                for name in test_data.obs_names:
                    match = re.match(r'(\d+\.\d+)x(\d+\.\d+)', name)
                    if match:
                        x, y = match.groups()
                        x_values.append(float(x))
                        y_values.append(float(y))

                # 将 x 和 y 存储到 obs 中的新列
                test_data.obs['x'] = x_values
                test_data.obs['y'] = y_values
                     
                
            if self.test_data_dir.endswith('.h5ad'):
                test_data = ad.read_h5ad(self.test_data_dir)

            # 判断是否使用ttest
            if self.bool_ttest == True:
                test_data = filter_adata_by_genes_data(test_data, self.ttest_genes_list_dir)

            print(f"Loaded test dataset from {self.test_data_dir}")
            return test_data
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            return None

    def load_simu_data(self):
        # 加载非ST-deconv生成的模拟数据
        file_name = f'{self.simu_data_dir}{self.simu_expression}'
        file_label = f'{self.simu_data_dir}{self.simu_label}'

        try:
            if file_name.endswith('.csv') or file_name.endswith('.tsv'):
                simu_data_df = pd.read_csv(file_name, sep='\t' if file_name.endswith('.tsv') else ',')

                # 数据转置
                simu_data_df = simu_data_df.transpose()
                simu_data_df.columns = simu_data_df.iloc[0]
                simu_data_df = simu_data_df.drop(simu_data_df.index[0])
                
                # 删除第一列并仅选择数值列
                simu_data_df_numeric = simu_data_df.values
                simu_data_df_numeric = np.array(simu_data_df, dtype=np.float64)
                
                # 将 DataFrame 转换为稀疏矩阵
                simu_data_csr = sp.csr_matrix(simu_data_df_numeric)
                
                # 创建 AnnData 对象并设置 X 为稀疏矩阵
                var_df = pd.DataFrame(index=simu_data_df.columns)
                
                # 将label作为obs加入anndata对象
                if file_label.endswith('.csv') or file_label.endswith('.tsv'):
                    simu_label_df = pd.read_csv(file_label, sep='\t' if file_label.endswith('.tsv') else ',')
                obs_df = pd.DataFrame(index=simu_data_df.index, data=simu_label_df.values, columns= simu_label_df.columns)
                simu_data = ad.AnnData(X=simu_data_csr, var=var_df, obs = obs_df)

                
            if self.simu_data_dir.endswith('.h5ad'):
                simu_data = ad.read_h5ad(self.simu_data_dir)

            print(f"Loaded simulated dataset from {self.simu_data_dir}")
            return simu_data
        except Exception as e:
            print(f"Error loading simulated dataset: {e}")
            return None

    def load_ttest_genes(self):
        # 加载ttest基因名列表
        try:
            ttest_genes = pd.read_csv(self.ttest_genes_list_dir, sep='\t', header=None).iloc[:, 0].tolist()
            print(f"Loaded ttest genes list from {self.ttest_genes_list_dir}")
            return ttest_genes
        except Exception as e:
            print(f"Error loading ttest genes list: {e}")
            return None
        
    def get_celltype_list(self):
        single_cell_data = self.load_single_cell_data()
        # 获取X矩阵的行数
        cell_type_matrix = np.array(single_cell_data.obs_names[:])
        # cell_type_matrix = np.array(single_cell_data.obs['cellType'])
        num_cells = cell_type_matrix.shape[0]

        # 创建一个二维列表用于已存储的细胞类型的值
        celltype_list = []

        # 定义需要并行遍历的细胞索引列表
        cell_indices = range(num_cells)

        for cell_idx in cell_indices:
            celltype_name = cell_type_matrix[cell_idx].split('.')[0]
            if celltype_name not in celltype_list:
                celltype_list.append(celltype_name)

        return celltype_list
    
# # 测试读取数据
# option = get_base_option_list()  # 将字符串解析为字典
# dataset = Dataset(option)
# dataset.load_single_cell_data()


