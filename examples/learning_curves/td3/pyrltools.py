import os
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
from evaluate_policy import evaluate_policy
import numpy as np

default_config = {
}

def env_factory_factory(config, **kwargs):
    def env_factory(**kwargs):
        env = gym.make(config["environment_name"], **kwargs)
        env = RescaleActionV0(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools
        env = gym.wrappers.ClipAction(env)
        env.reset(seed=config["seed"])
        return env
    return env_factory

def train_pyrltools(config, use_python_environment=True):
    custom_environment = {
        "path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../custom_environment"),
        "action_dim": 1,
        "observation_dim": 3,
    }
    env_factory = env_factory_factory(config)
    from pyrltools import TD3
    example_env = env_factory() 
    default_kwargs = {
        "ACTOR_TRAINING_INTERVAL": 2, 
        "CRITIC_TRAINING_INTERVAL": 1,
        "ACTOR_TARGET_UPDATE_INTERVAL": 2,
        "CRITIC_TARGET_UPDATE_INTERVAL": 2,
        "N_ENVIRONMENTS": 1,
    }
    kwargs = {
        **default_kwargs,
        "ACTOR_POLYAK": 1 - config["tau"],
        "CRITIC_POLYAK": 1 - config["tau"],
        "GAMMA": config["gamma"],
        "STEP_LIMIT": config["n_steps"],
        "ACTOR_BATCH_SIZE": config["batch_size"],
        "CRITIC_BATCH_SIZE": config["batch_size"],
        "OPTIMIZER_ALPHA": config["learning_rate"],
        "OPTIMIZER_EPSILON": 1e-8, # PyTorch default
        "ACTOR_HIDDEN_DIM": config["hidden_dim"],
        "CRITIC_HIDDEN_DIM": config["hidden_dim"],
        "N_WARMUP_STEPS": config["learning_starts"],
        "TARGET_NEXT_ACTION_NOISE_STD": config["target_next_action_noise_std"],
        "TARGET_NEXT_ACTION_NOISE_CLIP": config["target_next_action_noise_clip"],
        "EXPLORATION_NOISE": config["exploration_noise"],
    }
    interface_name = str(config["seed"])
    if use_python_environment:
        sac = TD3(env_factory, enable_evaluation=False, interface_name=interface_name, force_recompile=not "PYRLTOOLS_SKIP_FORCE_RECOMPILE" in os.environ, **kwargs)
    else:
        sac = TD3(custom_environment, interface_name=interface_name, **kwargs)
    state = sac.State(config["seed"])
    returns = []
    render = False
    for step_i in range(config["n_steps"]):
        if step_i % config["evaluation_interval"] == 0:
            current_returns = evaluate_policy(lambda observation: state.action(observation), config, env_factory, render=config["render"] and step_i >= 0)
            print(f"Step {step_i}/{config['n_steps']}: {np.mean(current_returns)}", flush=True)
            returns.append(current_returns)
        state.step()
    return returns
