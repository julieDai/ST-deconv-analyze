from ..dataset_config import Dataset
from ..options import set_override_value, get_option_list
import subprocess
import sys
from multiprocessing import Pool, Value
import os
import json
from itertools import product
import time
import logging
import GPUtil

# 定义要运行的模块列表
modules = [
    # 'ST-deconv.run.simu_data', #在创建新的并行进程的时候需要重新做预处理 所以放在了下面去
    'ST-deconv.run.train',
    # 'ST-deconv.run.finally_test'  
]

# 消融实验不同的覆盖情况
overrides_keys1 = [
    # 'AE',
    # 'AE_simu',
    'AE_spatial',
    # 'AE_ttest',
    # 'AE_DAN',
    # 'AE_simu_spatial',
    # 'AE_simu_ttest',
    # 'AE_simu_DAN',
    # 'AE_spatial_ttest',
    # 'AE_spatial_DAN',
    # 'AE_ttest_DAN',
    # 'AE_simu_spatial_ttest',
    # 'AE_simu_spatial_DAN',
    # 'AE_simu_ttest_DAN',
    # 'AE_spatial_ttest_DAN',
    # 'AE_simu_spatial_ttest_DAN'
]
overrides_trainset = [
    '01',
    # '02',
    # '03',
    # '04',
    # '05',
    # '31',
    # '32',
    # '33',
    # '34',
    # '35',
]

# result_file_name = input('input the result_file_name:')
result_file_name = 'trainModel_(30_30_60)_CLloss*0.01'


# 配置日志记录
logging.basicConfig(filename='process_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_key_and_modules(key_trainset, key):
    global last_used_gpu
    start_time = time.time()

    with last_used_gpu.get_lock():  # 确保线程安全
        last_used_gpu.value = (last_used_gpu.value + 1) % 2  # 有2个GPU
        last_used_gpu.value = 1 # 使用第2个GPU
        
        gpu_id = last_used_gpu.value

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    set_override_value(key, key_trainset)
    options = get_option_list(f'{result_file_name}_{key_trainset}') #单独训练每一个训练集
    
    # options = get_option_list(f'{result_file_name}') # 依次训练9个训练集
    
    env = os.environ.copy()
    env['OVERRIDE_OPTIONS'] = json.dumps(options)

    try:
        subprocess.run([sys.executable, '-m', 'ST-deconv.run.simu_data'], check=True, env=env)
        for module in modules:
            subprocess.run([sys.executable, '-m', module], check=True, env=env)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while executing {e.cmd}: {e.returncode}")
        logging.error(f"Output: {e.stdout}")
        logging.error(f"Errors: {e.stderr}")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Process on GPU {gpu_id}_{key_trainset}_{key} completed in {duration:.2f} seconds.")


if __name__ == '__main__':
    last_used_gpu = Value('i', -1)  # 初始化为-1，表示未使用任何GPU

    with Pool(4) as pool:
        combinations = list(product(overrides_trainset, overrides_keys1))
        
        pool.starmap(process_key_and_modules, combinations)
    