import random
import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from algorithms.agent import ConstrainedPPOAgent

from utils.buffers import Memory
from utils.colreg_handler import COLREGHandler

from colreg_logic import rtamt_yml_parser


# None - use the Unity Editor (press Play)
# "../Builds/train_gui.app"  - path to macos build
# "../Builds/train_5M.app" - path for 5M
unity_env_path = "../Builds/train.app"

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = "cpu"
# BoatAgent Parameters - must match those in Unity
OBSERVATION_SIZE = 20 # From UnityEnvironment/Scripts/BoatAgent.cs
RAYCAST_COUNT = 7 # 3 side rays + 1 front ray # From Unity RayPerceptionSensorComponent3D
RAYCAST_SIZE = RAYCAST_COUNT * 2 # Each ray (7) has a distance and a hit flag (1 or 0)
NUM_ROBUSTNESS_FLAG = 2 # R1, R2

INPUT_SIZE = OBSERVATION_SIZE + RAYCAST_SIZE + NUM_ROBUSTNESS_FLAG
ACTION_SIZE = 2 # Left Jet, Right Jet
BEHAVIOR_NAME = "BoatAgent"

ROLLOUT_SIZE = 2_048
TOT_STEPS = 2_048_000 # 1000 updates
GAMMA = 0.995
LR = 0.0003
BATCH_SIZE = 256
#BATCH_SIZE = 128
#ENTROPY_COEF = 0.0001
ENTROPY_COEF = 0.001
SAVE_INTERVAL = 20_480
START_SAFETY = TOT_STEPS // 2 # Activate safety constraints after roughly 50%, this number is a mupltiple of rollout size

colreg_path = "colreg_logic/colreg.yaml"

SAFE_DISTANCE = 2.0
NUM_EVAL_EPISODES = 10
EVAL_INTERVAL = 5 # Evaluate every 5 episodes during evaluation phase (after safety activation)

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

def evaluate_model(eval_seed, agent, colreg_handler, RTAMT):
    # Create a separate environment for evaluation
    engine_config = EngineConfigurationChannel()
    eval_env = UnityEnvironment(
        file_name=unity_env_path, 
        side_channels=[engine_config],
        seed=eval_seed,
        worker_id=eval_seed,
        no_graphics=False 
    )
    academy_parameters = {
        "seed": eval_seed,
    }
    eval_env._send_academy_parameters(academy_parameters)
    eval_env.reset()

    BEHAVIOR_NAME = list(eval_env.behavior_specs.keys())[0] 

    engine_config.set_configuration_parameters(width=800, height=600, time_scale=1.0)

    episodes_completed = 0
    total_rewards = []
    total_r1_robustness = []
    total_r2_robustness = []
    memory_buffer = Memory(stl_horizon=RTAMT.horizon_length)

    print(f"--- Starting Evaluation with seed {eval_seed} ---")

    try:
        while episodes_completed < NUM_EVAL_EPISODES:
            eval_env.reset() # Reset a inizio ciclo per pulizia
            decision_steps, terminal_steps = eval_env.get_steps(BEHAVIOR_NAME)
            episode_reward = 0.0
            done = False
            
            pbar = tqdm(desc=f"Episode {episodes_completed+1}/{NUM_EVAL_EPISODES}", unit="steps")
            
            while not done:
                obs, vec_obs = get_single_agent_obs(decision_steps)

                r1, r2 = memory_buffer.compute_markovian_flags()
                obs_augmented = np.concatenate((obs, [r1, r2]))
                obs_tensor = torch.from_numpy(obs_augmented).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action_tensor, _ = agent.get_action(obs_tensor, deterministic=True)
                
                action_numpy = action_tensor.cpu().numpy()
                #pbar.write(f"Step {pbar.n+1} | Throttle: {action_numpy[0,0]:.3f}, Steering: {action_numpy[0,1]:.3f} | flag_R1: {r1:.4f} | flag_R2: {r2:.4f}")
                action_tuple = ActionTuple()
                action_tuple.add_continuous(action_numpy)

                eval_env.set_actions(BEHAVIOR_NAME, action_tuple)
                eval_env.step()
                pbar.update(1)

                decision_steps, terminal_steps = eval_env.get_steps(BEHAVIOR_NAME)
                done = len(terminal_steps) > 0
                step_reward = float(terminal_steps.reward[0]) if done else float(decision_steps.reward[0])
                pbar.write(f"Step {pbar.n+1} | Step Reward: {step_reward:.5f} | flag_R1: {r1:.4f} | flag_R2: {r2:.4f}")
                episode_reward += step_reward

                r1_signal = colreg_handler.get_R1_safety_signal(obs=vec_obs, safe_dist=SAFE_DISTANCE)
                physical_speed = colreg_handler.get_ego_speed(vec_obs)
                memory_buffer.add_stl_sample(phys_speed=float(physical_speed), r1_signal=float(r1_signal))

            _ , single_rho = RTAMT.compute_robustness_dense(memory_buffer.stl_window)
            rho_1 = single_rho.get('R1_safe_distance', 0.0)
            rho_2 = single_rho.get('R2_safe_speed', 0.0)

            total_rewards.append(episode_reward)
            total_r1_robustness.append(rho_1)
            total_r2_robustness.append(rho_2)

            pbar.close()
            print(f"Episode {episodes_completed+1} finished | Return: {episode_reward:.2f} | R1: {rho_1:.2f} | R2: {rho_2:.2f}")

            memory_buffer.clear_stl_window()
            memory_buffer.clear_ppo() 
            episodes_completed += 1

    except KeyboardInterrupt:
        print("Evaluation manually interrupted.")
    finally:
        eval_env.close()
    
    return total_rewards, total_r1_robustness, total_r2_robustness

# final4 baseline setup: 
# gamma 0.995, lr 0.0003, ent 0.001, batchsize 256, logstd=0.0, gradclip 0.5 ( on critics too), unbound costs with scale 0.1
# smaller reward for facing target 1/5, SAFE_DISTANCE = 2.0, t_horizon=5.0 and RTAMT_horizon=80 (4s)
# testing also with new evaluating method
COST_SCALE =0.1 #1
def main():
    model_name = f"boat_agent_final4_NEWEVAL_GAMMA_{GAMMA}_lr_{LR}_ent_{ENTROPY_COEF}_batchsize_{BATCH_SIZE}_costscale_{COST_SCALE}"
    eval_seed = 31
    seeds= [1, 3, 7, 34, 42]
    seed_iteration = 0
    for seed in seeds:
        seed_iteration += 1
        print(f"--- Avvio Training Seed {seed} ({seed_iteration}/5) ---")
        set_all_seeds(seed)

        save_dir = f"Models/{model_name}/seed_{seed}"
        os.makedirs(save_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=f"runs/{model_name}/seed_{seed}")

        starting_step = 0
        n_updates = 0

        last_checkpoint_path = None
        best_safe_return_min = -float('inf')
        best_safe_return_mean = -float('inf')
        best_safe_return_pct = -float('inf')
        best_return = -float('inf')

        colreg_handler = COLREGHandler()

        RTAMT = rtamt_yml_parser.RTAMTYmlParser(colreg_path)

        # Channel used to speed up the game time
        engine_config = EngineConfigurationChannel()
    
        print("Loading environment...")
        env = UnityEnvironment(
            file_name=unity_env_path, 
            side_channels=[engine_config],
            worker_id=seed,
            seed=seed,
            no_graphics=False
        )

        env_academy_parameters = {
            "seed": seed,
        }

        env._send_academy_parameters(env_academy_parameters)

        env.reset()
        print("Environment loaded successfully.")
    
        # time_scale = 1.0 real time 20 step/s - 40.0 is 40x faster than real time 800 step/s
        engine_config.set_configuration_parameters(width=600, height=600, time_scale=40.0)

        # Debug info print behaviors available
        print("Behaviors found:", list(env.behavior_specs.keys()))
        behavior_name = list(env.behavior_specs.keys())[0] 
    
        agent = ConstrainedPPOAgent(
            INPUT_SIZE, 
            ACTION_SIZE, 
            device=DEVICE, 
            start_safety=START_SAFETY, 
            gamma=GAMMA,
            lr=LR,
            entropy_coeff=ENTROPY_COEF
        )

        if starting_step != 0:
            checkpoint_path = f"Models/{model_name}/seed_{seed}/steps_{starting_step}.pth"
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
        print(f"horizon length from RTAMT: {RTAMT.horizon_length}")
        try:
            s = starting_step
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            current_return = 0.0
            returns_episodes = []
            current_ep_cost_r1 = 0.0
            current_ep_cost_r2 = 0.0
            current_ep_pos_cost_r1 = 0.0
            current_ep_pos_cost_r2 = 0.0

            # Progress bar per il training
            pbar = tqdm(total=TOT_STEPS, desc=f"Training {seed_iteration}/5", unit="steps")

            save_model = False

            save_last_checkpoint = False

            window_size = 50 # Calculate average over the last 50 episodes
            recent_returns = deque(maxlen=window_size)
            recent_episode_cumulative_costs_r1 = deque(maxlen=window_size)
            recent_episode_cumulative_costs_r2 = deque(maxlen=window_size)
            recent_episode_pos_cumulative_costs_r1 = deque(maxlen=window_size)
            recent_episode_pos_cumulative_costs_r2 = deque(maxlen=window_size)
            min_episodes_to_evaluate = 10

            while s < TOT_STEPS: 

                mean_throttle_buffer, mean_steering_buffer = [], []
                std_throttle_buffer, std_steering_buffer = [], []
            
                while (len(memory_buffer.states) < ROLLOUT_SIZE):
                
                    obs, vec_obs = get_single_agent_obs(decision_steps)
                    r1_flag, r2_flag = memory_buffer.compute_markovian_flags()
                    obs_augmented = np.concatenate((obs, [r1_flag, r2_flag]))
                    obs_tensor = torch.from_numpy(obs_augmented).float().unsqueeze(0).to(DEVICE)

                    action_tensor, log_probabs = agent.get_action(obs_tensor)
                    action_numpy = action_tensor.detach().cpu().numpy()
                    action_tuple = ActionTuple()
                    action_tuple.add_continuous(action_numpy)

                    with torch.no_grad():
                        mean, _, std = agent.policy_net(obs_tensor)
                        mean_throttle_buffer.append(mean[0, 0].detach().cpu().numpy())
                        mean_steering_buffer.append(mean[0, 1].detach().cpu().numpy())
                        std_throttle_buffer.append(std[0, 0].detach().cpu().numpy())
                        std_steering_buffer.append(std[0, 1].detach().cpu().numpy())


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
                
                    rho_1 = single_rho.get('R1_safe_distance', 0.0)
                    rho_2 = single_rho.get('R2_safe_speed', 0.0)

                    #cost_1 = max(0, np.tanh(-rho_1)) 
                    #cost_2 = max(0, np.tanh(-rho_2))
                    cost_1 = np.tanh(-rho_1) * COST_SCALE
                    cost_2 = np.tanh(-rho_2) * COST_SCALE
                    pos_cost_1 = max(0, cost_1)
                    pos_cost_2 = max(0, cost_2)

                    memory_buffer.add_robustness(r1=rho_1,r2=rho_2)
                    memory_buffer.add_costs(c_r1=cost_1, c_r2=cost_2)

                    current_ep_cost_r1 += cost_1
                    current_ep_cost_r2 += cost_2
                    current_ep_pos_cost_r1 += pos_cost_1
                    current_ep_pos_cost_r2 += pos_cost_2

                    if end_episode:
                        recent_returns.append(current_return)
                        returns_episodes.append(current_return)
                        recent_episode_cumulative_costs_r1.append(current_ep_cost_r1)
                        recent_episode_cumulative_costs_r2.append(current_ep_cost_r2)
                        recent_episode_pos_cumulative_costs_r1.append(current_ep_pos_cost_r1)
                        recent_episode_pos_cumulative_costs_r2.append(current_ep_pos_cost_r2)
                        current_return = 0.0
                        current_ep_cost_r1 = 0.0
                        current_ep_cost_r2 = 0.0
                        current_ep_pos_cost_r1 = 0.0
                        current_ep_pos_cost_r2 = 0.0
                        memory_buffer.clear_stl_window()
                        env.reset()
                        decision_steps, terminal_steps = env.get_steps(behavior_name)

                    if s % SAVE_INTERVAL == 0:
                        save_model = True
                    
                    if s == START_SAFETY-1:
                        save_last_checkpoint = True
                    
                if save_last_checkpoint:
                    pre_safety_path = f"{save_dir}/pre_safety_checkpoint.pth"
                    torch.save(checkpoint, pre_safety_path)
                    pbar.write(f"Checkpoint saved before safety activation: {pre_safety_path}")
                    save_last_checkpoint = False


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
            
                log_dict = agent.update(rollouts=rollout_buffer, 
                                        robustness_dict=robustness_dict, 
                                        current_step=s, 
                                        batch_size=BATCH_SIZE,
                                         writer=writer)
                
                n_updates += 1
            
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

                if len(recent_returns) > 0:
                    smoothed_return = np.mean(recent_returns)
                    writer.add_scalar("Training/Smoothed_Return", smoothed_return, s)
                    writer.add_scalar("Training/Smoothed_Ep_Cost_R1", np.mean(recent_episode_cumulative_costs_r1), s)
                    writer.add_scalar("Training/Smoothed_Ep_Cost_R2", np.mean(recent_episode_cumulative_costs_r2), s)
                    writer.add_scalar("Training/Smoothed_Ep_Pos_Cost_R1", np.mean(recent_episode_pos_cumulative_costs_r1), s)
                    writer.add_scalar("Training/Smoothed_Ep_Pos_Cost_R2", np.mean(recent_episode_pos_cumulative_costs_r2), s)

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
                writer.add_scalar("Policy/Throttle_Mean", np.mean(mean_throttle_buffer), s)
                writer.add_scalar("Policy/Steering_Mean", np.mean(mean_steering_buffer), s)
                writer.add_scalar("Policy/Throttle_Std", np.mean(std_throttle_buffer), s)
                writer.add_scalar("Policy/Steering_Std", np.mean(std_steering_buffer), s)
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

                if n_updates % EVAL_INTERVAL == 0 and s >= START_SAFETY:

                    total_rewards, total_r1_robustness, total_r2_robustness = evaluate_model(eval_seed=eval_seed, agent=agent, colreg_handler=colreg_handler, RTAMT=RTAMT)

                    smooth_return = np.mean(total_rewards)

                    # Three different safety criteria
                    is_safe_min = min(total_r1_robustness) >= 0.0 and min(total_r2_robustness) >= 0.0
                    is_safe_mean = np.mean(total_r1_robustness) >= 0.0 and np.mean(total_r2_robustness) >= 0.0
                    
                    safe_pct_r1 = sum(1 for r1 in total_r1_robustness if r1 >= 0.0) / len(total_r1_robustness)
                    safe_pct_r2 = sum(1 for r2 in total_r2_robustness if r2 >= 0.0) / len(total_r2_robustness)
                    safety_threshold_percentage = 0.8
                    is_safe_pct = (safe_pct_r1 >= safety_threshold_percentage and 
                                   safe_pct_r2 >= safety_threshold_percentage)

                    # Save best model (no safety constraint)
                    if smooth_return > best_return:
                        best_return = smooth_return
                        best_model_path = f"{save_dir}/best_model.pth"
                        torch.save(checkpoint, best_model_path)
                        pbar.write(f"*** NEW BEST MODEL! Return: {best_return:.2f}, R1: {current_r1:.2f}, R2: {current_r2:.2f}     ***")

                    # Save best safe model (minimum criterion)
                    if is_safe_min and smooth_return > best_safe_return_min:
                        best_safe_return_min = smooth_return
                        best_safe_min_path = f"{save_dir}/best_safe_model_MIN.pth"
                        torch.save(checkpoint, best_safe_min_path)
                        pbar.write(f"*** NEW BEST SAFE MODEL (MIN)! Return: {best_safe_return_min:.2f} ***")

                    # Save best safe model (mean criterion)
                    if is_safe_mean and smooth_return > best_safe_return_mean:
                        best_safe_return_mean = smooth_return
                        best_safe_mean_path = f"{save_dir}/best_safe_model_MEAN.pth"
                        torch.save(checkpoint, best_safe_mean_path)
                        pbar.write(f"*** NEW BEST SAFE MODEL (MEAN)! Return: {best_safe_return_mean:.2f} ***")

                    # Save best safe model (percentage criterion)
                    if is_safe_pct and smooth_return > best_safe_return_pct:
                        best_safe_return_pct = smooth_return
                        best_safe_pct_path = f"{save_dir}/best_safe_model_PCT.pth"
                        torch.save(checkpoint, best_safe_pct_path)
                        pbar.write(f"*** NEW BEST SAFE MODEL (PCT={safety_threshold_percentage:.0%})! Return: {best_safe_return_pct:.2f} ***")

                    # Log all three metrics
                    writer.add_scalar("Eval/Safe_Min_R1", min(total_r1_robustness), s)
                    writer.add_scalar("Eval/Safe_Min_R2", min(total_r2_robustness), s)
                    writer.add_scalar("Eval/Safe_Mean_R1", np.mean(total_r1_robustness), s)
                    writer.add_scalar("Eval/Safe_Mean_R2", np.mean(total_r2_robustness), s)
                    writer.add_scalar("Eval/Safe_Pct_R1", safe_pct_r1, s)
                    writer.add_scalar("Eval/Safe_Pct_R2", safe_pct_r2, s)

        except KeyboardInterrupt:
            print("Manual interruption...")
            break

        finally:
            pbar.close()
            env.close()
            writer.close()
            print(f"Environment with seed {seed} closed.")
            time.sleep(5) 

if __name__ == "__main__":
    main()