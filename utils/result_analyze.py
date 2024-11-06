import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_rmse(rmse_split, rmse_othertest, folder_name, path_pic):
    plt.figure()
    plt.plot(rmse_split, label='RMSE Split', marker='o')
    plt.plot(rmse_othertest, label='RMSE OtherTest', marker='x')
    plt.xlabel('Index')
    plt.ylabel('RMSE Value')
    plt.title(f'RMSE Comparison for {folder_name}')
    plt.legend()
    plt.grid(True)

    # 确保路径存在
    os.makedirs(path_pic, exist_ok=True)

    # 保存图像
    plt.savefig(os.path.join(path_pic, f'{folder_name}.png'))
    plt.close()

def extract_rmse(file_path, keyword):
    rmse_values = []
    with open(file_path, 'r') as file:
        for line in file:
            if keyword in line:
                rmse_value = float(line.split(':')[1].strip())
                rmse_values.append(rmse_value)
    return rmse_values

def process_folder(folder_path, filename):
    result_rmse_path = os.path.join(folder_path, filename)    
    rmse_values = []
    
    # 提取 RMSE 值
    if os.path.exists(result_rmse_path):
        rmse = extract_rmse(result_rmse_path, 'rmse')
        other_rmse = extract_rmse(result_rmse_path, 'othertest_rmse')
        if rmse != []:
            rmse_values = rmse
        if other_rmse !=[]:
            rmse_values = other_rmse
    
    
    # 计算统计信息
    if rmse_values:
        rmse_mean = np.mean(rmse_values)
        rmse_max = np.max(rmse_values)
        rmse_min = np.min(rmse_values)
        rmse_median = np.median(rmse_values)
    else:
        rmse_mean = rmse_max = rmse_min = rmse_median = None
    
    return rmse_mean, rmse_max, rmse_min, rmse_median

def process_all_folders(base_folder, path_pic):
    results = []
    folder_names = []
    results_raw_split = []
    results_raw_othertest = []

    
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            rmse_split = extract_rmse(os.path.join(folder_path, 'result_rmse.txt') ,'rmse')
            rmse_othertest = extract_rmse(os.path.join(folder_path, 'result_other_rmse.txt'), 'othertest_rmse' )
            plot_rmse(rmse_split, rmse_othertest, folder_name, path_pic)

            rmse_mean, rmse_max, rmse_min, rmse_median = process_folder(folder_path, 'result_rmse.txt')
            other_rmse_mean, other_rmse_max, other_rmse_min, other_rmse_median = process_folder(folder_path, 'result_other_rmse.txt')
            results.append({
                'Folder': folder_name,
                'RMSE_Mean': rmse_mean,
                'RMSE_Max': rmse_max,
                'RMSE_Min': rmse_min,
                'RMSE_Median': rmse_median,
                'other_RMSE_Mean': other_rmse_mean,
                'other_RMSE_Max': other_rmse_max,
                'other_RMSE_Min': other_rmse_min,
                'other_RMSE_Median': other_rmse_median
            })
            folder_names.append(folder_name)
            results_raw_split.append(rmse_split)
            results_raw_othertest.append(rmse_othertest)

    
    return pd.DataFrame(results), pd.DataFrame(results_raw_split, index=folder_names).transpose(), pd.DataFrame(results_raw_othertest, index=folder_names).transpose()

# 示例调用
base_folder = '/home/daishurui/git_project/ST-deconv/data/experiment/CL4.0/trainModel_xiaorong_01'  # experience文件夹路径

print("输入experiment的文件夹名：")
base_filename = input()
base_filepath = os.path.join(base_folder, base_filename) 
print("输入不包含xslx后缀的文件名：")
file_name = input()

results_df, results_raw_split_df, results_raw_other_df = process_all_folders(base_filepath, f'rmse_summary/{file_name}')

with pd.ExcelWriter(f'./rmse_summary/{file_name}.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='analyze', index=False)
    results_raw_split_df.to_excel(writer, sheet_name='results_raw_split', index=False)
    results_raw_other_df.to_excel(writer, sheet_name='results_raw_other', index=False)

print("Summary saved to rmse_summary.csv")
