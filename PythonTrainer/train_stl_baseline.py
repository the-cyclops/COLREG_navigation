import random
import os
import time
import numpy as np
import torch
import itertools
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from algorithms.agent import ConstrainedPPOAgent
from stable_baselines3 import PPO

from utils.buffers import Memory
from utils.colreg_handler import COLREGHandler

from colreg_logic import rtamt_yml_parser

# None - use the Unity Editor (press Play)
# "../Builds/train_gui.app"  - path to macos build
# "../Builds/train_5M.app" - path for 5M
unity_env_path = None

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = "cpu"
# BoatAgent Parameters - must match those in Unity
OBSERVATION_SIZE = 24 # From UnityEnvironment/Scripts/BoatAgent.cs
RAYCAST_COUNT = 7 # 3 side rays + 1 front ray  From Unity RayPerceptionSensorComponent3D
RAYCAST_SIZE = RAYCAST_COUNT * 2 # Each ray (7) has a distance and a hit flag (1 or 0)
NUM_ROBUSTNESS_FLAG = 2 # R1, R2

INPUT_SIZE = OBSERVATION_SIZE + RAYCAST_SIZE + NUM_ROBUSTNESS_FLAG
GAMMA = 0.99
ACTION_SIZE = 2 # Left Jet, Right Jet
BEHAVIOR_NAME = "BoatAgent"

ROLLOUT_SIZE = 2_048
TOT_STEPS = 1_024_000# 500 updates

SAVE_INTERVAL = 20_480
START_SAFETY = TOT_STEPS +1 # Stay in reward-only for parameter tuning

colreg_path = "colreg_logic/colreg.yaml"

log_dir = "runs/stl_baseline_grid_search/"

SAFE_DISTANCE = 1.0

# Hyperparameter Grid
LEARNING_RATES = [3e-5, 1e-4, 3e-4]
ENTROPY_COEFS = [0.0, 0.0001, 0.001]
FIXED_SEED = 42 # Keep seed fixed for fair comparison between hyperparameters

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_single_agent_obs(steps):
    # Extract raw observations list
    raw_obs = steps.obs
    
    # Check shapes to determine sensor order and extract agent 0 immediately
    if raw_obs[0].shape[1] == RAYCAST_SIZE and raw_obs[1].shape[1] == OBSERVATION_SIZE:
        ray_obs = raw_obs[0][0]
        vec_obs = raw_obs[1][0]
    elif raw_obs[0].shape[1] == OBSERVATION_SIZE and raw_obs[1].shape[1] == RAYCAST_SIZE:
        ray_obs = raw_obs[1][0]
        vec_obs = raw_obs[0][0]
    else:
        raise ValueError(f"Unexpected shapes: {raw_obs[0].shape}, {raw_obs[1].shape}")
    
    # Concatenate to get a 1D array
    return np.concatenate((ray_obs, vec_obs)), vec_obs

def main():
    model_name = f"Grid_Search_DifferentialNormalized_gamma_{GAMMA}_only_distance_and_target_reward" # For saving models and TensorBoard logs

    hp_combinations = list(itertools.product(LEARNING_RATES, ENTROPY_COEFS))
    total_runs = len(hp_combinations)
    
    for run_idx, (lr, entropy) in enumerate(hp_combinations, 1):
        
        run_name = f"lr_{lr}_ent_{entropy}"
        print(f"\n--- Starting Training Run ({run_idx}/{total_runs}) | LR: {lr}, Entropy: {entropy} ---")
        
        set_all_seeds(FIXED_SEED)

        save_dir = f"Models/{model_name}/{run_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Channel used to speed up the game time
        engine_config = EngineConfigurationChannel()
    
        print("Loading environment...")
        env = UnityEnvironment(
            file_name=unity_env_path, 
            side_channels=[engine_config],
            seed=FIXED_SEED,
            no_graphics=False
        )

        env = UnityToGymWrapper(env, allow_multiple_obs=True)

        env.reset()
        print("Environment loaded and wrapped successfully.")
    
        # time_scale = 1.0 real time 20 step/s - 40.0 is 40x faster than real time 800 step/s
        engine_config.set_configuration_parameters(width=800, height=600, time_scale=40.0)

        # Ensure ConstrainedPPOAgent __init__ accepts lr and entropy_coeff
        # agent = ConstrainedPPOAgent(
        #     INPUT_SIZE, 
        #     ACTION_SIZE, 
        #     device=DEVICE, 
        #     start_safety=START_SAFETY, 
        #     gamma=GAMMA,
        #     lr=lr,
        #     entropy_coeff=entropy 
        # )
        agent = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=lr,
            ent_coef=entropy,
            gamma=GAMMA, 
            verbose=1, 
            seed=42,
            tensorboard_log=log_dir+f"{model_name}/{run_name}",
            device=DEVICE
        )

        try:
            agent.learn(total_timesteps=TOT_STEPS, tb_log_name=run_name, reset_num_timesteps=True)
        except KeyboardInterrupt:
            print("\nManual interrupt received. Saving model before exiting...")
            agent.save(f"{save_dir}/manual_interrupt_model")

        finally:
            env.close()
            print(f"Environment with LR: {lr}, Entropy: {entropy} closed. Moving to next hyperparameter combination.")
            time.sleep(5)

if __name__ == "__main__":
    main()