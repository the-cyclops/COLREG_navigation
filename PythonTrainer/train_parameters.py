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

from algorithms.agent import ConstrainedPPOAgent

from utils.buffers import Memory
from utils.colreg_handler import COLREGHandler

from colreg_logic import rtamt_yml_parser

model_name = "Grid_Search_hardcap_std"
# None - use the Unity Editor (press Play)
# "../Builds/train_gui.app"  - path to macos build
# "../Builds/train_5M.app" - path for 5M
unity_env_path = "../Builds/emptyscene.app"

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = "cpu"
# BoatAgent Parameters - must match those in Unity
OBSERVATION_SIZE = 24 # From UnityEnvironment/Scripts/BoatAgent.cs
RAYCAST_COUNT = 7 # 3 side rays + 1 front ray  From Unity RayPerceptionSensorComponent3D
RAYCAST_SIZE = RAYCAST_COUNT * 2 # Each ray (7) has a distance and a hit flag (1 or 0)
NUM_ROBUSTNESS_FLAG = 2 # R1, R2

INPUT_SIZE = OBSERVATION_SIZE + RAYCAST_SIZE + NUM_ROBUSTNESS_FLAG
GAMMA = 0.995
ACTION_SIZE = 2 # Left Jet, Right Jet
BEHAVIOR_NAME = "BoatAgent"

ROLLOUT_SIZE = 2_048
TOT_STEPS = 1_024_000# 500 updates

SAVE_INTERVAL = 20_480
START_SAFETY = TOT_STEPS +1 # Stay in reward-only for parameter tuning

colreg_path = "colreg_logic/colreg.yaml"

SAFE_DISTANCE = 1.0

# Hyperparameter Grid
LEARNING_RATES = [1e-4, 3e-4, 5e-4]
ENTROPY_COEFS = [0.001, 0.005, 0.01]
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

    hp_combinations = list(itertools.product(LEARNING_RATES, ENTROPY_COEFS))
    total_runs = len(hp_combinations)
    
    for run_idx, (lr, entropy) in enumerate(hp_combinations, 1):
        
        run_name = f"lr_{lr}_ent_{entropy}"
        print(f"\n--- Starting Training Run ({run_idx}/{total_runs}) | LR: {lr}, Entropy: {entropy} ---")
        
        set_all_seeds(FIXED_SEED)

        save_dir = f"Models/{model_name}/{run_name}"
        os.makedirs(save_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=f"runs/{model_name}/{run_name}")

        starting_step = 0

        last_checkpoint_path = None
        best_feasible_return = -float('inf')

        colreg_handler = COLREGHandler()

        RTAMT = rtamt_yml_parser.RTAMTYmlParser(colreg_path)

        safety_active = False

        # Channel used to speed up the game time
        engine_config = EngineConfigurationChannel()
    
        print("Loading environment...")
        env = UnityEnvironment(
            file_name=unity_env_path, 
            side_channels=[engine_config],
            seed=FIXED_SEED,
            no_graphics=False
        )
    
        env.reset()
        print("Environment loaded successfully.")
    
        # time_scale = 1.0 real time 20 step/s - 40.0 is 40x faster than real time 800 step/s
        engine_config.set_configuration_parameters(width=800, height=600, time_scale=40.0)

        # Debug info print behaviors available
        print("Behaviors found:", list(env.behavior_specs.keys()))
        behavior_name = list(env.behavior_specs.keys())[0] 
    
        # Ensure ConstrainedPPOAgent __init__ accepts lr and entropy_coeff
        agent = ConstrainedPPOAgent(
            INPUT_SIZE, 
            ACTION_SIZE, 
            device=DEVICE, 
            start_safety=START_SAFETY, 
            gamma=GAMMA,
            lr=lr,
            entropy_coeff=entropy 
        )

        if starting_step != 0:
            checkpoint_path = f"Models/{model_name}/{run_name}/steps_{starting_step}.pth"
            print(f"Loading model from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            starting_step = checkpoint['step']
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.value_net.load_state_dict(checkpoint['value_state_dict'])
            agent.cost_net_safe_distance.load_state_dict(checkpoint['cost_net_safe_distance_state_dict'])
            agent.cost_net_safe_speed.load_state_dict(checkpoint['cost_net_safe_speed_state_dict'])
            agent.policy_opt.load_state_dict(checkpoint['policy_opt_state_dict'])
            agent.value_opt.load_state_dict(checkpoint['value_opt_state_dict'])
            agent.cost_opts[0].load_state_dict(checkpoint['cost_safe_distance_opt_state_dict'])
            agent.cost_opts[1].load_state_dict(checkpoint['cost_safe_speed_opt_state_dict'])
            print(f"Model loaded, starting from step {starting_step}.")
        else:
            print(f"Start training on: {behavior_name}")

        memory_buffer = Memory(stl_horizon=RTAMT.horizon_length)

        try:
            s = starting_step
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            current_return = 0.0
            returns_episodes = []

            # Training progress bar
            pbar = tqdm(total=TOT_STEPS, desc=f"Run {run_idx}/{total_runs} [LR:{lr} Ent:{entropy}]", unit="steps")

            save_model = False

            while s < TOT_STEPS: 

                if not safety_active and s >= START_SAFETY:
                    safety_active = True
                    pbar.write(f"Safety constraints activated at step {s}.")

                # TEMPORARY
                mean_buffer = []
                std_buffer = []
            
                while (len(memory_buffer.states) < ROLLOUT_SIZE):
                
                    obs, vec_obs = get_single_agent_obs(decision_steps)
                    r1, r2 = memory_buffer.compute_markovian_flags()
                    obs_augmented = np.concatenate((obs, [r1, r2]))
                    obs_tensor = torch.from_numpy(obs_augmented).float().unsqueeze(0).to(DEVICE)

                    action_tensor, log_probabs = agent.get_action(obs_tensor)
                    action_numpy = action_tensor.detach().cpu().numpy()
                    action_tuple = ActionTuple()
                    action_tuple.add_continuous(action_numpy)

                    # TEMPORARY
                    with torch.no_grad():
                        mean, _, std = agent.policy_net(obs_tensor)
                        mean_buffer.append(mean.detach().cpu().numpy())
                        std_buffer.append(std.detach().cpu().numpy())

                    env.set_actions(behavior_name, action_tuple)
                    env.step()
                    s += 1
                    pbar.update(1)

                    decision_steps, terminal_steps = env.get_steps(behavior_name)
                    end_episode = len(terminal_steps) > 0

                    reward = float(terminal_steps.reward[0]) if end_episode else float(decision_steps.reward[0])
                    current_return += reward

                    memory_buffer.add_ppo_transition(
                        state=obs_tensor, 
                        action=action_tensor, 
                        logprob=log_probabs,
                        reward=reward, 
                        is_terminal=float(end_episode)
                    )
                
                    r1_signal = colreg_handler.get_R1_safety_signal(obs=vec_obs, safe_dist=SAFE_DISTANCE)

                    physical_speed = colreg_handler.get_ego_speed(vec_obs)

                    memory_buffer.add_stl_sample(phys_speed=float(physical_speed), r1_signal=float(r1_signal))
                

                    _ , single_rho = RTAMT.compute_robustness_dense(memory_buffer.stl_window)
                
                    #rho_1 = single_rho.get('R1_safe_distance', 0.0)
                    #rho_2 = single_rho.get('R2_safe_speed', 0.0)
                    rho_1, rho_2 = 1.0, 1.0
                    cost_1 = max(0, -rho_1) / SAFE_DISTANCE
                    cost_2 = max(0, -rho_2)
                    memory_buffer.add_robustness(r1=rho_1, r2=rho_2)
                    memory_buffer.add_costs(c_r1=cost_1, c_r2=cost_2)

                    if end_episode:
                        returns_episodes.append(current_return)
                        current_return = 0.0
                        memory_buffer.clear_stl_window()
                        env.reset()
                        decision_steps, terminal_steps = env.get_steps(behavior_name)

                    if s % SAVE_INTERVAL == 0:
                        save_model = True

                next_state = get_single_agent_obs(decision_steps)[0]
                r1_next, r2_next = memory_buffer.compute_markovian_flags()
                next_state_augmented = np.concatenate((next_state, [r1_next, r2_next]))
                
                rollout_buffer = {}
                rollout_buffer['states'] =  memory_buffer.states
                rollout_buffer['actions'] = memory_buffer.actions
                rollout_buffer['logprobs'] = memory_buffer.logprobs
                rollout_buffer['rewards'] = np.array(memory_buffer.rewards)
                rollout_buffer['masks'] = 1 - np.array(memory_buffer.is_terminals)
                rollout_buffer['next_state'] = np.array(next_state_augmented)
                rollout_buffer['cost_r1'] = np.array(memory_buffer.cost_r1)
                rollout_buffer['cost_r2'] = np.array(memory_buffer.cost_r2)

                robustness_dict = {'R1': min(memory_buffer.robustness_1), 'R2': min(memory_buffer.robustness_2)}
            
                log_dict = agent.update_with_minibatches(rollouts=rollout_buffer, robustness_dict=robustness_dict, current_step=s)
            
                mode = log_dict['mode']

                rewards = rollout_buffer['rewards']
                gae_returns = log_dict['reward'][1]

                pbar.write(f"----- Update! Mode: {mode} -----\n Reward: {rewards.mean().item():.4f} | GAE_returns: {gae_returns.mean().item():.4f} | Rho R1: {robustness_dict['R1']:.4f} | Rho R2: {robustness_dict['R2']:.4f}") 
                
                mean_return = None
                if returns_episodes:
                    mean_return = np.mean(returns_episodes)
                    pbar.write(f"Mean Return: {mean_return:.2f}")
                    writer.add_scalar("Training/Mean_Return", mean_return, s)
                    returns_episodes.clear()

                pbar.set_postfix({
                    'Reward': f"{rewards.mean().item():.2f}",
                    'R1': f"{robustness_dict['R1']:.2f}",
                    'R2': f"{robustness_dict['R2']:.2f}"
                })

                writer.add_scalar("Training/Mean_Reward", rewards.mean().item(), s)
                writer.add_scalar("Training/Value_target_mean_GAE_returns", gae_returns.mean().item(), s)
                writer.add_scalar("Training/Robustness_R1_Physics", robustness_dict['R1'], s)
                writer.add_scalar("Training/Robustness_R2_Physics", robustness_dict['R2'], s)
                writer.add_scalar("Training/R1_GAE_cumulative_cost", log_dict['r1'][1].mean().item(), s)
                writer.add_scalar("Training/R2_GAE_cumulative_cost", log_dict['r2'][1].mean().item(), s)
                writer.add_text("Training/Mode_Log", mode, s)
                writer.add_scalar("Policy/Policy_Mean", np.mean(mean_buffer), s)
                writer.add_scalar("Policy/Policy_Std", np.mean(std_buffer), s)
                writer.add_scalar("Policy/Entropy", log_dict['entropy'], s)
                writer.add_scalar("Loss/Policy", log_dict['policy_loss'], s)
                writer.add_scalar("Loss/Value", log_dict['value_loss'], s)
                writer.add_scalar("Loss/Cost_R1", log_dict['cost_loss_r1'], s)
                writer.add_scalar("Loss/Cost_R2", log_dict['cost_loss_r2'], s)

                memory_buffer.clear_ppo()

                # Save the model occasionally
                checkpoint = {
                        'step': s,
                        'policy_state_dict': agent.policy_net.state_dict(),
                        'value_state_dict': agent.value_net.state_dict(),
                        'cost_net_safe_distance_state_dict': agent.cost_net_safe_distance.state_dict(),
                        'cost_net_safe_speed_state_dict': agent.cost_net_safe_speed.state_dict(),
                        'policy_opt_state_dict': agent.policy_opt.state_dict(),
                        'value_opt_state_dict': agent.value_opt.state_dict(),
                        'cost_safe_distance_opt_state_dict': agent.cost_opts[0].state_dict(),
                        'cost_safe_speed_opt_state_dict': agent.cost_opts[1].state_dict(),
                        'robustness_r1': robustness_dict['R1'],
                        'robustness_r2': robustness_dict['R2']
                    }
                if save_model:
                    current_path = f"{save_dir}/steps_{s}.pth"
                    torch.save(checkpoint, current_path)
                    pbar.write(f"Checkpoint saved: {current_path}")

                    if last_checkpoint_path is not None and last_checkpoint_path != current_path:
                        if os.path.exists(last_checkpoint_path):
                            try:
                                os.remove(last_checkpoint_path)
                            except OSError as e:
                                pbar.write(f"Warning: could not delete old checkpoint: {e}")
                    
                    last_checkpoint_path = current_path
                    save_model = False

                current_r1 = robustness_dict['R1']
                current_r2 = robustness_dict['R2']             

                is_feasible = (current_r1 >= 0.0) and (current_r2 >= 0.0)
                if mean_return is not None and s >= START_SAFETY:
                    if is_feasible and (mean_return > best_feasible_return):
                        best_feasible_return = mean_return
                        best_path = f"{save_dir}/best_feasible_model.pth"
                    
                        # Save specific copy
                        torch.save(checkpoint, best_path)
                        pbar.write(f"*** NEW BEST FEASIBLE MODEL! Return: {best_feasible_return:.2f}, R1: {current_r1:.2f}, R2: {current_r2:.2f}    ***")

        except KeyboardInterrupt:
            print("Manual interruption... Exiting the entire grid search.")
            # Break completely out of the function so it doesn't just jump to the next hyperparameter set
            return

        finally:
            pbar.close()
            env.close()
            writer.close()
            print(f"Environment for LR: {lr}, Entropy: {entropy} closed.")
            time.sleep(5) 

if __name__ == "__main__":
    main()