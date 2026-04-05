import random
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from time import sleep
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from algorithms.agent import ConstrainedPPOAgent
from utils.buffers import Memory
from utils.colreg_handler import COLREGHandler
from colreg_logic import rtamt_yml_parser

# --- CONFIGURATIONS ---
model_name = "boat_agent_final4_NEWEVAL_GAMMA_0.995_lr_0.0003_ent_0.001_batchsize_256_costscale_0.1/seed_34"
unity_env_path = None #"../Builds/emptyscene.app" 
DEVICE = "cpu"
OBSERVATION_SIZE = 20
RAYCAST_COUNT = 7 
RAYCAST_SIZE = RAYCAST_COUNT * 2 
NUM_ROBUSTNESS_FLAG = 2 
INPUT_SIZE = OBSERVATION_SIZE + RAYCAST_SIZE + NUM_ROBUSTNESS_FLAG
ACTION_SIZE = 2 

colreg_path = "colreg_logic/colreg.yaml"
SAFE_DISTANCE = 2.0
NUM_EVAL_EPISODES = 10 
FIXED_SEED = 402

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    set_all_seeds(FIXED_SEED)
    
    #checkpoint_path = f"Models/{model_name}/pre_safety_checkpoint.pth"
    #checkpoint_path = f"Models/{model_name}/best_model.pth"
    #checkpoint_path = f"Models/{model_name}/best_safe_model.pth"
    checkpoint_path = f"Models/{model_name}/best_safe_model_MEAN.pth"
    #checkpoint_path = f"Models/{model_name}/best_safe_model_PCT.pth"
    #checkpoint_path = f"Models/{model_name}/steps_2048000.pth"
    
    print(f"--- Starting Evaluation from model: {checkpoint_path} ---")
    
    colreg_handler = COLREGHandler()
    RTAMT = rtamt_yml_parser.RTAMTYmlParser(colreg_path)
    memory_buffer = Memory(stl_horizon=RTAMT.horizon_length)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found at path: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    print(f"Loaded checkpoint from {checkpoint_path} from step {checkpoint['step']}")
    sleep(3)

    engine_config = EngineConfigurationChannel()
    env_params = EnvironmentParametersChannel()
    env_params.set_float_parameter("eval_episode_seed", -1.0)
    
    env = UnityEnvironment(
        file_name=unity_env_path, 
        side_channels=[engine_config, env_params],
        seed=FIXED_SEED,
        no_graphics=False 
    )
    env.reset()
    
    BEHAVIOR_NAME = list(env.behavior_specs.keys())[0]
    print(f"Connected to behavior: {BEHAVIOR_NAME}")

    engine_config.set_configuration_parameters(width=800, height=600, time_scale=5.0)

    agent = ConstrainedPPOAgent(INPUT_SIZE, ACTION_SIZE, device=DEVICE, start_safety=0)
    
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.value_net.load_state_dict(checkpoint['value_state_dict'])
    agent.cost_net_safe_distance.load_state_dict(checkpoint['cost_net_safe_distance_state_dict'])
    agent.cost_net_safe_speed.load_state_dict(checkpoint['cost_net_safe_speed_state_dict'])
    
    agent.policy_net.eval()
    agent.value_net.eval()
    agent.cost_net_safe_distance.eval()
    agent.cost_net_safe_speed.eval()

    total_rewards = []
    total_r1_robustness = []
    total_r2_robustness = []

    try:
        for ep in range(NUM_EVAL_EPISODES):
            current_seed = FIXED_SEED + ep
            env_params.set_float_parameter("eval_episode_seed", float(current_seed))
            env.reset() 
            decision_steps, terminal_steps = env.get_steps(BEHAVIOR_NAME)
            episode_reward = 0.0
            done = False
            
            pbar = tqdm(desc=f"Episode {ep+1}/{NUM_EVAL_EPISODES}", unit="steps")
            
            while not done:
                obs, vec_obs = get_single_agent_obs(decision_steps)

                r1_flag, r2_flag = memory_buffer.compute_markovian_flags()
                obs_augmented = np.concatenate((obs, [r1_flag, r2_flag]))
                obs_tensor = torch.from_numpy(obs_augmented).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action_tensor, _ = agent.get_action(obs_tensor, deterministic=True)
                
                action_numpy = action_tensor.cpu().numpy()
                action_tuple = ActionTuple()
                action_tuple.add_continuous(action_numpy)

                env.set_actions(BEHAVIOR_NAME, action_tuple)
                env.step()
                pbar.update(1)

                decision_steps, terminal_steps = env.get_steps(BEHAVIOR_NAME)
                done = len(terminal_steps) > 0
                step_reward = float(terminal_steps.reward[0]) if done else float(decision_steps.reward[0])
                episode_reward += step_reward

                r1_signal = colreg_handler.get_R1_safety_signal(obs=vec_obs, safe_dist=SAFE_DISTANCE)
                physical_speed = colreg_handler.get_ego_speed(vec_obs)
                memory_buffer.add_stl_sample(phys_speed=float(physical_speed), r1_signal=float(r1_signal))

                # Calcolo RTAMT per lo step corrente
                _ , single_rho = RTAMT.compute_robustness_dense(memory_buffer.stl_window)
                step_r1 = single_rho.get('R1_safe_distance', 0.0)
                step_r2 = single_rho.get('R2_safe_speed', 0.0)

                # SALVATAGGIO: Conserva lo storico dell'episodio nel buffer
                memory_buffer.add_robustness(r1=step_r1, r2=step_r2)

                # STAMPA: Mostriamo le flag Markoviane che l'agente ha effettivamente visto in questo step
                pbar.write(f"Step Reward: {step_reward:.2f} | Flag R1: {r1_flag:.4f} | Flag R2: {r2_flag:.4f}")

            # ESTRAZIONE VERO MINIMO: Cerca il valore peggiore registrato in tutto l'episodio
            rho_1 = min(memory_buffer.robustness_1) if memory_buffer.robustness_1 else 0.0
            rho_2 = min(memory_buffer.robustness_2) if memory_buffer.robustness_2 else 0.0

            total_rewards.append(episode_reward)
            total_r1_robustness.append(rho_1)
            total_r2_robustness.append(rho_2)

            pbar.close()
            print(f"Episode {ep+1} finished | Return: {episode_reward:.2f} | Min R1: {rho_1:.4f} | Min R2: {rho_2:.4f}")

            memory_buffer.clear_stl_window()
            memory_buffer.clear_ppo() 

    except KeyboardInterrupt:
        print("Evaluation manually interrupted.")
    finally:
        env_params.set_float_parameter("eval_episode_seed", -1.0)
        env.close()
        if total_rewards:
            print("\n--- Final Evaluation Results ---")
            print(f"Average Return: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
            print(f"Min R1 Robustness: {np.min(total_r1_robustness):.2f}")
            print(f"Min R2 Robustness: {np.min(total_r2_robustness):.2f}")
            print(f"Mean R1 Robustness: {np.mean(total_r1_robustness):.2f}")
            print(f"Mean R2 Robustness: {np.mean(total_r2_robustness):.2f}")
        else:
            print("\nNo episodes completed.")

if __name__ == "__main__":
    main()