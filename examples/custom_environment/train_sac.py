from rltools import SAC
import os, sys, argparse
from multiprocessing import Process
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_utils import load_json_results, write_to_csv, generate_sobol_samples, SAC_Search_Space

parser = argparse.ArgumentParser(description='Start generating data')
parser.add_argument('--num_configs', type=int, default=512, help='Number of configurations to generate')
parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to generate per configuration')
parser.add_argument('--max_processes', type=int, default=1, help='Number of processes to run in parallel')
parser.add_argument('--seed', type=int, default=24, help='Seed for sobol samples generation')

def train(params, seed, env):

    sac = SAC(env,
            evaluation_interval=1000,
            num_evaluation_episodes=10,
            GAMMA=params["GAMMA"],
            ACTOR_BATCH_SIZE=params["BATCH_SIZE"],
            CRITIC_BATCH_SIZE=params["BATCH_SIZE"],
            CRITIC_TRAINING_INTERVAL=params["CRITIC_TRAINING_INTERVAL"],
            ACTOR_TRAINING_INTERVAL=params["ACTOR_TRAINING_INTERVAL"],
            CRITIC_TARGET_UPDATE_INTERVAL=params["CRITIC_TARGET_UPDATE_INTERVAL"],
            ACTOR_POLYAK=params["ACTOR_POLYAK"],
            CRITIC_POLYAK=params["CRITIC_POLYAK"],
            N_ENVIRONMENTS=1,
            STEP_LIMIT=10000,
            ACTOR_HIDDEN_DIM=params["ACTOR_HIDDEN_DIM"],
            ACTOR_NUM_LAYERS=params["ACTOR_NUM_LAYERS"],
            ACTOR_ACTIVATION_FUNCTION=params["ACTOR_ACTIVATION_FUNCTION"],
            CRITIC_HIDDEN_DIM=params["CRITIC_HIDDEN_DIM"],
            CRITIC_NUM_LAYERS=params["CRITIC_NUM_LAYERS"],
            CRITIC_ACTIVATION_FUNCTION=params["CRITIC_ACTIVATION_FUNCTION"],
            ALPHA=params["ALPHA"],
            LOG_STD_LOWER_BOUND=params["LOG_STD_LOWER_BOUND"],
            LOG_STD_UPPER_BOUND=params["LOG_STD_UPPER_BOUND"],
            OPTIMIZER_ALPHA=params["OPTIMIZER_ALPHA"],
            )
    state = sac.State(seed)
    state.set_exp_name("pendulum", "sac")

    finished = False
    while not finished:
        finished = state.step()

    path = state.export_return()
    json_path = os.path.join(path, "return.json")
    data = load_json_results(json_path)

    header = ['GAMMA', 
              'BATCH_SIZE',
              'CRITIC_TRAINING_INTERVAL',
              'ACTOR_TRAINING_INTERVAL',
              'CRITIC_TARGET_UPDATE_INTERVAL',
              'ACTOR_POLYAK',
              'CRITIC_POLYAK',
              'ACTOR_HIDDEN_DIM',
              'ACTOR_NUM_LAYERS',
              'ACTOR_ACTIVATION_FUNCTION',
              'CRITIC_HIDDEN_DIM',
              'CRITIC_NUM_LAYERS',
              'CRITIC_ACTIVATION_FUNCTION',
              'ALPHA',
              'LOG_STD_LOWER_BOUND',
              'LOG_STD_UPPER_BOUND',
              'OPTIMIZER_ALPHA',
              'returns_mean',
              'returns_std',
              'episode_length_mean',
              'episode_length_std',
              'seed',
              'env',
              'algo'
            ]
    write_to_csv(header, seed, 'pendulum', 'sac', params, data[-1], 'pendulum_sac.csv')

    return data

if __name__ == "__main__":
    args = parser.parse_args()

    custom_environment = {
        "path": os.path.dirname(os.path.abspath(__file__)),
        "action_dim": 1,
        "observation_dim": 3,
    }

    num_configs = args.num_configs
    num_samples = args.num_samples
    max_processes = args.max_processes
    seed = args.seed

    configs = generate_sobol_samples(num_configs, SAC_Search_Space, seed)
    seeds = [i for i in range(num_samples)]

    processes = []
    for i in range(num_configs):
        for j in range(num_samples):
            params = configs[i]
            seed = seeds[j]
            p = Process(target=train, args=(params, seed, custom_environment))
            processes.append(p)

    step = 0
    while step < len(processes):
        begin = int(step * max_processes)
        end = min(begin + max_processes, len(processes))
        for p in processes[begin:end]:
            p.start()
        for p in processes[begin:end]:
            p.join()
        step += 1