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

# Define the list of modules to run
modules = [
    # 'ST-deconv.run.simu_data', # Preprocessing must be redone when creating new parallel processes, so it's moved below
    'ST-deconv.run.train',
    # 'ST-deconv.run.finally_test'  
]

# Ablation experiment with different coverage scenarios
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


# Configure logging
logging.basicConfig(filename='process_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_key_and_modules(key_trainset, key):
    global last_used_gpu
    start_time = time.time()

    with last_used_gpu.get_lock():  # Ensure thread safety
        last_used_gpu.value = (last_used_gpu.value + 1) % 2  # 有2个GPU
        last_used_gpu.value = 1 # Use the second GPU
        
        gpu_id = last_used_gpu.value

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    set_override_value(key, key_trainset)
    options = get_option_list(f'{result_file_name}_{key_trainset}') # Train each dataset separately
    
    # options = get_option_list(f'{result_file_name}') # Sequentially train all 9 datasets
    
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
    last_used_gpu = Value('i', -1)  # Initialize to -1, meaning no GPU has been used

    with Pool(4) as pool:
        combinations = list(product(overrides_trainset, overrides_keys1))
        
        pool.starmap(process_key_and_modules, combinations)
    
