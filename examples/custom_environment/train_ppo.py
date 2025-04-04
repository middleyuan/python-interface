from rltools import PPO
import os, sys, argparse
from multiprocessing import Process
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_utils import load_json_results, write_to_csv, generate_sobol_samples, PPO_Search_Space

parser = argparse.ArgumentParser(description='Start generating data')
parser.add_argument('--num_configs', type=int, default=512, help='Number of configurations to generate')
parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to generate per configuration')
parser.add_argument('--max_processes', type=int, default=1, help='Number of processes to run in parallel')
parser.add_argument('--seed', type=int, default=24, help='Seed for sobol samples generation')

def train(params, seed, env):

    ppo = PPO(env,
            evaluation_interval=100,
            num_evaluation_episodes=10,
            GAMMA=params["GAMMA"],
            LAMBDA=params["LAMBDA"],
            EPSILON_CLIP=params["EPSILON_CLIP"],
            INITIAL_ACTION_STD=params["INITIAL_ACTION_STD"],
            ACTION_ENTROPY_COEFFICIENT=params["ACTION_ENTROPY_COEFFICIENT"],
            ADVANTAGE_EPSILON=params["ADVANTAGE_EPSILON"],
            POLICY_KL_EPSILON=params["POLICY_KL_EPSILON"],
            N_WARMUP_STEPS_CRITIC=params["N_WARMUP_STEPS_CRITIC"],
            N_WARMUP_STEPS_ACTOR=params["N_WARMUP_STEPS_ACTOR"],
            N_EPOCHS=params["N_EPOCHS"],
            STEP_LIMIT=2000,
            ACTOR_HIDDEN_DIM=params["ACTOR_HIDDEN_DIM"],
            ACTOR_NUM_LAYERS=params["ACTOR_NUM_LAYERS"],
            ACTOR_ACTIVATION_FUNCTION=params["ACTOR_ACTIVATION_FUNCTION"],
            CRITIC_HIDDEN_DIM=params["CRITIC_HIDDEN_DIM"],
            CRITIC_NUM_LAYERS=params["CRITIC_NUM_LAYERS"],
            CRITIC_ACTIVATION_FUNCTION=params["CRITIC_ACTIVATION_FUNCTION"],
            N_ENVIRONMENTS=params["N_ENVIRONMENTS"],
            ON_POLICY_RUNNER_STEPS_PER_ENV=params["ON_POLICY_RUNNER_STEPS_PER_ENV"],
            BATCH_SIZE=params["BATCH_SIZE"],
            OPTIMIZER_ALPHA=params["OPTIMIZER_ALPHA"],
            )
    state = ppo.State(seed)
    state.set_exp_name("pendulum", "ppo")

    finished = False
    while not finished:
        finished = state.step()

    path = state.export_return()
    json_path = os.path.join(path, "return.json")
    data = load_json_results(json_path)

    header = ['GAMMA', 
              'LAMBDA', 
              'EPSILON_CLIP', 
              'INITIAL_ACTION_STD', 
              'ACTION_ENTROPY_COEFFICIENT', 
              'ADVANTAGE_EPSILON', 
              'POLICY_KL_EPSILON', 
              'N_WARMUP_STEPS_CRITIC', 
              'N_WARMUP_STEPS_ACTOR', 
              'N_EPOCHS', 
              'ACTOR_HIDDEN_DIM', 
              'ACTOR_NUM_LAYERS', 
              'ACTOR_ACTIVATION_FUNCTION', 
              'CRITIC_HIDDEN_DIM', 
              'CRITIC_NUM_LAYERS', 
              'CRITIC_ACTIVATION_FUNCTION', 
              'ON_POLICY_RUNNER_STEPS_PER_ENV', 
              'BATCH_SIZE', 
              'OPTIMIZER_ALPHA',
              'returns_mean',
              'returns_std',
              'episode_length_mean',
              'episode_length_std',
              'seed',
              'env',
              'algo'
            ]
    write_to_csv(header, seed, 'pendulum', 'ppo', params, data[-1], 'pendulum_ppo.csv')

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

    configs = generate_sobol_samples(num_configs, PPO_Search_Space, seed)
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