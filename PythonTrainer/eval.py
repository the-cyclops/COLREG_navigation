import random
import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from algorithms.agent import ConstrainedPPOAgent

from utils.buffers import Memory
from utils.colreg_handler import COLREGHandler

from colreg_logic import rtamt_yml_parser

# --- CONFIGURATIONS ---
model_name = "boat_agent_model_v2_5M_steps"
seed = 42 # Choose the seed to evaluate
checkpoint_path = f"Models/{model_name}/seed_{seed}/best_feasible_model.pth"
unity_env_path = "../Builds/train_5M.app" 

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = "cpu"
# BoatAgent Parameters - must match those in Unity
OBSERVATION_SIZE = 24 # From UnityEnvironment/Scripts/BoatAgent.cs
RAYCAST_COUNT = 7 # 3 side rays + 1 front ray # From Unity RayPerceptionSensorComponent3D
RAYCAST_SIZE = RAYCAST_COUNT * 2 # Each ray (7) has a distance and a hit flag (1 or 0)
NUM_ROBUSTNESS_FLAG = 2 # R1, R2

INPUT_SIZE = OBSERVATION_SIZE + RAYCAST_SIZE + NUM_ROBUSTNESS_FLAG

ACTION_SIZE = 2 
BEHAVIOR_NAME = "BoatAgent"

colreg_path = "colreg_logic/colreg.yaml"
SAFE_DISTANCE = 1.0

NUM_EVAL_EPISODES = 10 # How many test episodes you want to run

def get_single_agent_obs(steps):
    raw_obs = steps.obs
    if raw_obs[0].shape[1] == RAYCAST_SIZE and raw_obs[1].shape[1] == OBSERVATION_SIZE:
        ray_obs = raw_obs[0][0]
        vec_obs = raw_obs[1][0]
    elif raw_obs[0].shape[1] == OBSERVATION_SIZE and raw_obs[1].shape[1] == RAYCAST_SIZE:
        ray_obs = raw_obs[1][0]
        vec_obs = raw_obs[0][0]
    else:
        raise ValueError(f"Unexpected shapes: {raw_obs[0].shape}, {raw_obs[1].shape}")
    return np.concatenate((ray_obs, vec_obs)), vec_obs

def main():
    print(f"--- Starting Evaluation from model: {checkpoint_path} ---")

    # Initialize modules
    colreg_handler = COLREGHandler()
    RTAMT = rtamt_yml_parser.RTAMTYmlParser(colreg_path)
    memory_buffer = Memory(stl_horizon=RTAMT.horizon_length)

    # Initialize environment
    engine_config = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=unity_env_path, 
        side_channels=[engine_config],
        seed=seed,
        no_graphics=False # Important to see the behavior
    )
    env.reset()
    
    # Set speed to real-time (1.0)
    engine_config.set_configuration_parameters(width=800, height=600, time_scale=1.0)

    print("Environment initialized successfully!")

    # Initialize and load the agent
    agent = ConstrainedPPOAgent(INPUT_SIZE, ACTION_SIZE, device=DEVICE, start_safety=0)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found at path: {checkpoint_path}")
    
    checkpoint_path = f"Models/{model_name}/seed_{seed}/steps_{starting_step}.pth"
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    starting_step = checkpoint['step']
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.value_net.load_state_dict(checkpoint['value_state_dict'])
    agent.cost_net_safe_distance.load_state_dict(checkpoint['cost_net_safe_distance_state_dict'])
    agent.cost_net_safe_speed.load_state_dict(checkpoint['cost_net_safe_speed_state_dict'])
    
    agent.policy_net.eval()
    agent.value_net.eval()
    agent.cost_net_safe_distance.eval()
    agent.cost_net_safe_speed.eval()

    print("Model loaded successfully!")

    episodes_completed = 0
    total_rewards = []
    total_r1_robustness = []
    total_r2_robustness = []

    try:
        decision_steps, terminal_steps = env.get_steps(BEHAVIOR_NAME)
        
        while episodes_completed < NUM_EVAL_EPISODES:
            episode_reward = 0.0
            done = False
            
            pbar = tqdm(desc=f"Episode {episodes_completed+1}/{NUM_EVAL_EPISODES}", unit="steps")
            
            while not done:
                obs, vec_obs = get_single_agent_obs(decision_steps)
                
                # Calculation of Markovian flags (necessary in eval too because they are part of the state)
                r1, r2 = memory_buffer.compute_markovian_flags()
                obs_augmented = np.concatenate((obs, [r1, r2]))
                obs_tensor = torch.from_numpy(obs_augmented).float().unsqueeze(0).to(DEVICE)

                # No gradient calculation during evaluation
                with torch.no_grad():
                    # NOTE: if your get_action returns a sampled action, during evaluation 
                    # it would be better to extract the mean (deterministic action) if the architecture allows it.
                    action_tensor, _ = agent.get_action(obs_tensor,deterministic=True)
                
                action_numpy = action_tensor.cpu().numpy()
                action_tuple = ActionTuple()
                action_tuple.add_continuous(action_numpy)

                env.set_actions(BEHAVIOR_NAME, action_tuple)
                env.step()
                pbar.update(1)

                decision_steps, terminal_steps = env.get_steps(BEHAVIOR_NAME)
                done = len(terminal_steps) > 0

                # Get reward
                step_reward = float(terminal_steps.reward[0]) if done else float(decision_steps.reward[0])
                episode_reward += step_reward

                # Update the STL buffer for the next calculation of r1, r2
                r1_signal = colreg_handler.get_R1_safety_signal(obs=vec_obs, safe_dist=SAFE_DISTANCE)
                physical_speed = colreg_handler.get_ego_speed(vec_obs)
                memory_buffer.add_stl_sample(phys_speed=float(physical_speed), r1_signal=float(r1_signal))

            # End of Episode: Calculate the overall STL robustness of the episode
            _ , single_rho = RTAMT.compute_robustness_dense(memory_buffer.stl_window)
            rho_1 = single_rho.get('R1_safe_distance', 0.0)
            rho_2 = single_rho.get('R2_safe_speed', 0.0)

            total_rewards.append(episode_reward)
            total_r1_robustness.append(rho_1)
            total_r2_robustness.append(rho_2)

            pbar.close()
            print(f"Episode {episodes_completed+1} finished | Reward: {episode_reward:.2f} | R1: {rho_1:.2f} | R2: {rho_2:.2f}")

            # Reset for the next episode
            memory_buffer.clear_stl_window()
            memory_buffer.clear_ppo() # Clear internal lists if necessary
            env.reset()
            decision_steps, terminal_steps = env.get_steps(BEHAVIOR_NAME)
            episodes_completed += 1

    except KeyboardInterrupt:
        print("Evaluation manually interrupted.")
    
    finally:
        env.close()
        print("\n--- Final Evaluation Results ---")
        print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average R1 Robustness: {np.mean(total_r1_robustness):.2f}")
        print(f"Average R2 Robustness: {np.mean(total_r2_robustness):.2f}")

if __name__ == "__main__":
    main()