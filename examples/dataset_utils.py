import json
import csv
import fcntl
import os, sys
import numpy as np
import re
from scipy.stats import qmc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_spaces import PPO_Search_Space, SAC_Search_Space, TD3_Search_Space

def load_json_results(file_path):
    """Load training results from a JSON file"""
    with open(file_path, "r") as f:
        content = f.read()
        content = re.sub(r"-?nan", "null", content, flags=re.IGNORECASE)

    data = json.loads(content)
    data = none_check(data)
    return data

def none_check(data):
    """Collapse data if it contains None values in backward manner"""
    for d in reversed(data):
        for metric in ['returns_mean', 'returns_std', 'episode_length_mean', 'episode_length_std']:
            if d[metric] is None:
                data.remove(d)
                break
    return data

def write_to_csv(header, seed, env, algo, params, data, csv_file):
    """Safely writes data to a shared CSV file with a lock mechanism."""
    with open(csv_file, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            if os.stat(csv_file).st_size == 0 and header:
                writer.writerow(header)
            
            row = []
            for h in header:
                if h in params:
                    row.append(params[h])
                elif h in data:
                    row.append(data[h])
                elif h == 'seed':
                    row.append(seed)
                elif h == 'env':
                    row.append(env)
                elif h == 'algo':
                    row.append(algo)
                else:
                    raise ValueError(f"Header {h} not found in params or data.")
                
            writer.writerow(row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def get_domain(search_space):
    """Get bounds based on search_space (dict)"""
    bound_list = []
    for key, value in search_space.items():
        if key == 'categorical':
            for k, v in value.items():
                bound_list.append([v[0], v[-1]])
        elif key == 'float':
            for k, v in value.items():
                bound_list.append([v[0], v[-1]])
    return np.array(bound_list).T

def transform_to_valid_config(configs, search_space):
    """Transform config to valid config based on search_space"""
    
    config_copy = configs.copy()
    if config_copy.ndim == 1:
        config_copy = config_copy.reshape(1, -1)
    batch_size = config_copy.shape[0]
    indx = np.zeros_like(config_copy, dtype=int)
    
    result = {}
    i = 0
    for key, value in search_space.items():
        if key == 'categorical':
            for k, v in value.items():
                v_array = np.array(v)
                for j in range(batch_size):
                    indx[j, i] = np.argmin(np.abs(v_array - config_copy[j, i]))
                    config_copy[j, i] = v_array[indx[j, i]]
                result[k] = config_copy[:, i]
                i += 1
        elif key == 'float':
            for k, v in value.items():
                result[k] = config_copy[:, i]
                i += 1
    
    return config_copy, indx

def transform_to_args(configs, search_space, indx):
    """Transform config to args based on search_space"""
    
    config_copy = configs.copy()
    if config_copy.ndim == 1:
        config_copy = config_copy.reshape(1, -1)
    batch_size = config_copy.shape[0]
    hp_configs = [{} for _ in range(batch_size)]
    
    i = 0
    for key, value in search_space.items():
        if key == 'categorical':
            for k, v in value.items():
                for j in range(batch_size):
                    if k in ['ACTOR_ACTIVATION_FUNCTION', 'CRITIC_ACTIVATION_FUNCTION']:
                        hp_configs[j][k] = ['RELU', 'TANH', 'FAST_TANH', 'SIGMOID'][indx[j, i]]
                    else:
                        hp_configs[j][k] = v[indx[j, i]]
                i += 1
        elif key == 'float':
            for k, v in value.items():
                for j in range(batch_size):
                    hp_configs[j][k] = config_copy[j, i]
                i += 1
    return hp_configs

def generate_sobol_samples(num_samples, search_space, seed):
    """Generate Sobol samples for the parameter space"""
    dim = len(search_space['categorical']) + len(search_space['float'])
    sampler = qmc.Sobol(d=dim, seed=seed)
    samples = sampler.random(num_samples)
    
    bounds = get_domain(search_space)
    configs = qmc.scale(samples, bounds[0], bounds[1])
    
    valid_configs, indx = transform_to_valid_config(configs, search_space)
    hp_configs = transform_to_args(valid_configs, search_space, indx)
    
    return hp_configs

if __name__ == "__main__":
 
    # Test get_domain
    print('Search Space:')
    print(get_domain(PPO_Search_Space))
    print(get_domain(SAC_Search_Space))
    print(get_domain(TD3_Search_Space))
    
    # Test pipeline
    configs = generate_sobol_samples(2, PPO_Search_Space, seed=0)
    print('Generated Hyperparameter Configs:')
    print(configs)
    configs = generate_sobol_samples(2, SAC_Search_Space, seed=0)
    print('Generated Hyperparameter Configs:')
    print(configs)
    configs = generate_sobol_samples(2, TD3_Search_Space, seed=0)
    print('Generated Hyperparameter Configs:')
    print(configs)