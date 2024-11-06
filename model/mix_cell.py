import os
import numpy as np
import time
import random
random.seed(42)
from scipy.sparse import csr_matrix, vstack

def generate_random_ratio(total_rows, celltype_list):
    """
    为给定的细胞类型列表生成一系列随机比率字典。

    每个字典包含细胞类型作为键，其对应的比率作为值。所有比率的和等于1。

    参数:
    - total_rows (int): 总行数或数据点数。
    - celltype_list (list): 细胞类型名称的列表。

    返回:
    - ratios (list): 包含每个数据点随机比率的字典列表。
    """
    ratios = []  # 初始化比率列表
    n = len(celltype_list)  # 细胞类型的数量

    for i in range(total_rows):
        ratio = {}  # 每个数据点的比率字典
        total_sum = 0  # 已分配的比率总和

        # 为每个细胞类型（除最后一个之外）生成随机比率
        for j in range(n - 1):
            # 生成一个随机比率，并确保所有比率之和不超过1
            ratio[celltype_list[j]] = random.uniform(0, 1 - total_sum)
            total_sum += ratio[celltype_list[j]]  # 更新已分配的比率总和

        # 为最后一个细胞类型计算比率，确保所有比率之和恰好为1
        last_ratio = 1 - total_sum
        ratio[celltype_list[n - 1]] = last_ratio

        ratios.append(ratio)  # 将当前比率字典添加到列表中
    print("generate_random_ratio finished")


    return ratios

def select_rows_by_ratio(data_list: dict, ratio, sample_num, merge_row):
    """
    根据给定的比率从每种细胞类型中选择行。

    参数:
    - data_list (dict): 一个字典，键为细胞类型，值为相应细胞类型的单细胞数据索引列表。
    - ratio (list): 包含每个细胞类型比率的字典列表，每个字典对应一个数据点的细胞类型比率。
    - merge_rows (int): 总共需要合并的行数。

    返回:
    - spot_rows (list): 一个列表，包含根据比率选择的行的索引。
    """
    spot_rows = []  # 初始化最终选定行的列表

    # 遍历每个比率字典，每个字典代表一个数据点的比率配置
    for i in range(len(ratio)):
        selected_rows = []  # 初始化当前数据点选定的行索引列表

        # 遍历每种细胞类型及其数据
        for cell_type, data in data_list.items():
            # 根据细胞类型的比率和merge_rows计算需要选择的行数
            num_selected = int(sample_num * ratio[i][cell_type] * merge_row)

            # 允许重复选择数据，从给定细胞类型的数据中随机选择指定数量的行
            selected_rows.extend(random.choices(data, k=num_selected))

        # 将当前数据点选定的行索引列表添加到最终列表中
        spot_rows.append(selected_rows)
    print("select_rows_by_ratio finished")

    return spot_rows

class Mix:
    def __init__(self, single_cell_data, celltype_list, sample_num, merge_rows, filepath):
        # 初始化Mix类，处理和准备基因表达数据集

        # 储存基因表达矩阵和细胞类型标签
        self.gene_expression_matrix = single_cell_data.X
        self.cell_type_matrix = np.array(single_cell_data.obs_names[:]) # h5ad
        # self.cell_type_matrix = np.array(single_cell_data.obs[:])  # csv读取单细胞转录组数据

        # 储存其他参数，包括细胞类型列表，样本数，合并行数，以及文件路径
        self.cell_type_list = celltype_list
        self.sample_num = sample_num
        self.merge_rows = merge_rows
        self.filepath = os.path.join(filepath, 'one_type_to_gene_index')
        os.makedirs(self.filepath, exist_ok=True)

        # 执行关键函数来创建细胞类型索引并获取随机细胞索引
        self.is_cell_type_index = self.get_cell_type_index()
        self.ratio = generate_random_ratio(self.merge_rows, self.cell_type_list)
        self.random_cell_index = self.get_random_cell()
        print("Mix __init__ finished")


    def get_cell_type_index(self):
        # 获取每种细胞类型的索引数组
        start_time = time.time()

        for cell_type in self.cell_type_list:
            print(f'Current For: {cell_type}')
            # 使用列表推导式查找属于当前细胞类型的索引
            celltype_indices = [idx for idx, name in enumerate(self.cell_type_matrix)
                                if name.split('.')[0] == cell_type]  # h5ad

            # celltype_indices = [idx for idx, name in enumerate(self.cell_type_matrix)
            #                     if name == cell_type]  # csv读取单细胞转录组数据
            elapsed_time = time.time() - start_time
            print(f"Processed {cell_type}, shape:{len(celltype_indices)}. "
                  f"Elapsed time: {elapsed_time:.2f} seconds.")

            # 保存每种细胞类型的索引到.npy文件
            filename = f"{cell_type.replace('/', '&')}_to_gene.npy"
            np.save(os.path.join(self.filepath, filename), np.array(celltype_indices))
        print("get_cell_type_index finished")
            

        return "completed index create"

    def get_random_cell(self):
        # 根据比例选择随机细胞行的索引
        data_list = {}
        for cell_type in self.cell_type_list:
            filename = f"{cell_type.replace('/', '&')}_to_gene.npy"
            # 加载之前保存的每种细胞类型的索引
            cell_array = np.load(os.path.join(self.filepath, filename))
            data_list[cell_type] = cell_array.tolist()

        # 使用给定的比例和merge_rows选择随机细胞行的索引
        random_cell_index = select_rows_by_ratio(data_list, self.ratio, self.sample_num, self.merge_rows)
        print("get_random_cell finished")

        return random_cell_index
    
    def gain_index_matrix(self):
        merge_rows = self.merge_rows
        # 预先分配一个列表来存储每个样本的总和
        rows_list = []
        for i in range(merge_rows):
            # 直接计算所选行的总和
            selected_rows_sum = np.sum(self.gene_expression_matrix[self.random_cell_index[i], :].toarray(), axis=0)
            # 将结果转换为CSR格式并添加到列表中
            rows_list.append(csr_matrix(selected_rows_sum))

            if i % 10 == 0:
                print(f"Processed {i} merged.")

        # 使用vstack一次性堆叠所有行
        random_cell_csr = vstack(rows_list)
        print("gain_index_matrix finished")

        return random_cell_csr