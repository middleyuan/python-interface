import os
from rltools import PPO, color
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

env = gym.make("Pendulum-v1")
seed = 0x1337
def env_factory(**kwargs):
    env = gym.make("Pendulum-v1", **kwargs)
    env = RescaleAction(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools
    
    env.reset(seed=seed)
    return env

custom_environment = {
    "path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "custom_environment"),
    "action_dim": 1,
    "observation_dim": 3,
}


for env_name, selected_env in [("RLtools", custom_environment), ("Gymnasium", env_factory)]:
    print(f"Training using the {color(env_name, 'red')} environment")
    ppo = PPO(
        selected_env,
        interface_name=env_name,
        enable_evaluation=True,
        evaluation_interval=100,
        # force_blas=True,
        num_evaluation_episodes=10,
        OPTIMIZER_ALPHA=1e-4,
        ACTION_ENTROPY_COEFFICIENT = 0.0,
        N_EPOCHS = 1,
        GAMMA = 0.9,
        BATCH_SIZE = 128,
        ACTOR_HIDDEN_DIM = 32,
        CRITIC_HIDDEN_DIM = 32,
        ON_POLICY_RUNNER_STEPS_PER_ENV = 128,
        N_ENVIRONMENTS = 8,
        STEP_LIMIT = 2000, # 1024 * 4 * 74 ~ 300k steps
    )
    state = ppo.State(seed)

    # Training
    finished = False
    while not finished:
        finished = state.step()

    # Save Checkpoint (so it can be loaded by inference.py)
    with open("pendulum_ppo_checkpoint.h", "w") as f:
        f.write(state.export_policy())

# Inference
env_replay = env_factory(render_mode="human")
env_replay.reset(seed=seed)

while True:
    observation, _ = env_replay.reset()
    finished = False
    current_return = 0
    while not finished:
        env_replay.render()
        action = state.action(observation)
        observation, reward, terminated, truncated, _ = env_replay.step(action)
        current_return += reward
        finished = terminated or truncated
    print(f"Return: {current_return}")