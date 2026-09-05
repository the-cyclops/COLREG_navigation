import random
import os
import time
import itertools
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from algorithms.agent import ConstrainedPPOAgent
from utils.buffers import Memory
from utils.colreg_handler import COLREGHandler
from colreg_logic import rtamt_yml_parser

# Unity build path
unity_env_path = "../Builds/empty_scene.app"
DEVICE = "cpu"

# Parametri ambiente Unity
OBSERVATION_SIZE = 20
RAYCAST_COUNT = 7
RAYCAST_SIZE = RAYCAST_COUNT * 2
NUM_ROBUSTNESS_FLAG = 3  # R1, R2, R6

INPUT_SIZE = OBSERVATION_SIZE + RAYCAST_SIZE + NUM_ROBUSTNESS_FLAG
ACTION_SIZE = 2
BEHAVIOR_NAME = "BoatAgent"

ROLLOUT_SIZE = 2_048
TOT_STEPS = 512_000 # 250 updates
SAVE_INTERVAL = 20_480
START_SAFETY = TOT_STEPS + 1  # Reward-only durante il tuning
COST_SCALE = 0.1
REWARD_SCALE = 0.1

# Hyperparameter Grid
GAMMAS = [0.99, 0.995]
LEARNING_RATES = [1e-4, 3e-4]
ENTROPY_COEFS = [0.0001, 0.001]
BATCH_SIZES = [64, 128, 256]
FIXED_SEED = 420

DUMMY_FLAGS = np.zeros(NUM_ROBUSTNESS_FLAG, dtype=np.float32)
DUMMY_ROBUSTNESS = {'R1': 1.0, 'R2': 1.0, 'R6': 1.0}

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
    model_name = "GRID_SEARCH_EMPTY_SCENE"
    hp_combinations = list(itertools.product(LEARNING_RATES, ENTROPY_COEFS, BATCH_SIZES, GAMMAS))
    total_runs = len(hp_combinations)

    for run_idx, (lr, entropy, batch_size, gamma) in enumerate(hp_combinations, 1):
        run_name = f"GAMMA_{gamma}_lr_{lr}_ent_{entropy}_batchsize_{batch_size}"
        print(f"\n--- Run ({run_idx}/{total_runs}) | LR: {lr}, Ent: {entropy}, Batch: {batch_size}, Gamma: {gamma} ---")

        set_all_seeds(FIXED_SEED)

        save_dir = f"Models/{model_name}/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=f"runs/{model_name}/{run_name}")

        last_checkpoint_path = None
        best_return = -float('inf')

        engine_config = EngineConfigurationChannel()
        env_params = EnvironmentParametersChannel()
        env_params.set_float_parameter("seed", float(FIXED_SEED))
        env_params.set_float_parameter("is_eval_scene", 0.0)

        env = UnityEnvironment(
            file_name=unity_env_path,
            side_channels=[engine_config, env_params],
            worker_id=FIXED_SEED + run_idx,
            seed=FIXED_SEED,
            no_graphics=False
        )
        env.reset()
        engine_config.set_configuration_parameters(width=600, height=600, time_scale=40.0)
        behavior_name = list(env.behavior_specs.keys())[0]

        agent = ConstrainedPPOAgent(
            INPUT_SIZE,
            ACTION_SIZE,
            device=DEVICE,
            start_safety=START_SAFETY,
            gamma=gamma,
            lr=lr,
            entropy_coeff=entropy
        )
        agent.set_train_mode()

        memory_buffer = Memory(tau=80)
        recent_returns = deque(maxlen=50)
        returns_episodes = []

        try:
            s = 0
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            current_return = 0.0
            save_model = False

            pbar = tqdm(total=TOT_STEPS, desc=f"Run {run_idx}/{total_runs}", unit="steps")

            while s < TOT_STEPS:
                mean_throttle_buf, mean_steer_buf = [], []
                std_throttle_buf, std_steer_buf = [], []

                # Raccolta pura a dimensione fissa (nessuna attesa di fine episodio)
                while len(memory_buffer.states) < ROLLOUT_SIZE:
                    obs, vec_obs = get_single_agent_obs(decision_steps)
                    obs_augmented = np.concatenate((obs, DUMMY_FLAGS))
                    obs_tensor = torch.from_numpy(obs_augmented).float().unsqueeze(0).to(DEVICE)

                    action_tensor, log_probabs = agent.get_action(obs_tensor)
                    action_numpy = action_tensor.detach().cpu().numpy()
                    action_tuple = ActionTuple()
                    action_tuple.add_continuous(action_numpy)

                    with torch.no_grad():
                        mean, _, std = agent.policy_net(obs_tensor)
                        mean_throttle_buf.append(mean[0, 0].cpu().item())
                        mean_steer_buf.append(mean[0, 1].cpu().item())
                        std_throttle_buf.append(std[0, 0].cpu().item())
                        std_steer_buf.append(std[0, 1].cpu().item())

                    env.set_actions(behavior_name, action_tuple)
                    env.step()
                    s += 1
                    pbar.update(1)

                    decision_steps, terminal_steps = env.get_steps(behavior_name)
                    end_episode = len(terminal_steps) > 0

                    reward = float(terminal_steps.reward[0]) if end_episode else float(decision_steps.reward[0])
                    reward *= REWARD_SCALE
                    current_return += reward

                    # Passiamo 0.0 per i segnali fisici non usati nell'arena vuota
                    memory_buffer.add_ppo_transition(
                        state=obs_tensor,
                        action=action_tensor,
                        logprob=log_probabs,
                        reward=reward,
                        is_terminal=float(end_episode),
                        phys_speed=0.0,
                        r1_signal=0.0,
                        keep_signal=0.0,
                        no_turn_signal=0.0
                    )

                    if end_episode:
                        recent_returns.append(current_return)
                        returns_episodes.append(current_return)
                        current_return = 0.0
                        env.reset()
                        decision_steps, terminal_steps = env.get_steps(behavior_name)

                    if s % SAVE_INTERVAL == 0:
                        save_model = True

                next_obs, next_vec_obs = get_single_agent_obs(decision_steps)
                next_state_augmented = np.concatenate((next_obs, DUMMY_FLAGS))
                num_samples = len(memory_buffer.states)

                rollout_buffer = {
                    'states': memory_buffer.states,
                    'actions': memory_buffer.actions,
                    'logprobs': memory_buffer.logprobs,
                    'rewards': np.array(memory_buffer.rewards),
                    'masks': 1 - np.array(memory_buffer.is_terminals),
                    'next_state': np.array(next_state_augmented),
                    'cost_r1': np.zeros(num_samples, dtype=np.float32),
                    'cost_r2': np.zeros(num_samples, dtype=np.float32),
                    'cost_r6': np.zeros(num_samples, dtype=np.float32)
                }

                log_dict = agent.update(
                    rollouts=rollout_buffer,
                    robustness_dict=DUMMY_ROBUSTNESS,
                    current_step=s,
                    batch_size=batch_size,
                    writer=writer
                )

                rewards = rollout_buffer['rewards']
                mean_return = np.mean(returns_episodes) if returns_episodes else None

                pbar_dict = {'Rew': f"{rewards.mean().item():.2f}"}
                if mean_return is not None:
                    pbar_dict['MeanRet'] = f"{mean_return:.1f}"
                    writer.add_scalar("Training/Mean_Return", mean_return, s)
                    returns_episodes.clear()

                pbar.set_postfix(pbar_dict)

                if len(recent_returns) > 0:
                    writer.add_scalar("Training/Smoothed_Return", np.mean(recent_returns), s)

                writer.add_scalar("Training/Mean_Reward", rewards.mean().item(), s)
                writer.add_scalar("Policy/Throttle_Mean", np.mean(mean_throttle_buf), s)
                writer.add_scalar("Policy/Steering_Mean", np.mean(mean_steer_buf), s)
                writer.add_scalar("Policy/Entropy", log_dict['entropy'], s)
                writer.add_scalar("Loss/Policy", log_dict['policy_loss'], s)
                writer.add_scalar("Loss/Value", log_dict['value_loss'], s)

                memory_buffer.clear_ppo()

                checkpoint = {
                    'step': s,
                    'policy_state_dict': agent.policy_net.state_dict(),
                    'value_state_dict': agent.value_net.state_dict(),
                    'cost_net_safe_distance_state_dict': agent.cost_net_safe_distance.state_dict(),
                    'cost_net_safe_speed_state_dict': agent.cost_net_safe_speed.state_dict(),
                    'cost_net_r6_state_dict': agent.cost_net_R6.state_dict(),
                    'policy_opt_state_dict': agent.policy_opt.state_dict(),
                    'value_opt_state_dict': agent.value_opt.state_dict(),
                    'cost_safe_distance_opt_state_dict': agent.cost_opts[0].state_dict(),
                    'cost_safe_speed_opt_state_dict': agent.cost_opts[1].state_dict(),
                    'cost_safe_r6_opt_state_dict': agent.cost_opts[2].state_dict(),
                    'robustness_r1': 1.0,
                    'robustness_r2': 1.0,
                    'robustness_r6': 1.0
                }

                if save_model:
                    current_path = f"{save_dir}/steps_{s}.pth"
                    torch.save(checkpoint, current_path)
                    if last_checkpoint_path and os.path.exists(last_checkpoint_path):
                        os.remove(last_checkpoint_path)
                    last_checkpoint_path = current_path
                    save_model = False

                if mean_return is not None and mean_return > best_return:
                    best_return = mean_return
                    torch.save(checkpoint, f"{save_dir}/best_model.pth")

        except KeyboardInterrupt:
            print("Interruzione manuale del grid search.")
            env.close()
            writer.close()
            return

        finally:
            pbar.close()
            env.close()
            writer.close()
            time.sleep(3)

if __name__ == "__main__":
    main()