import numpy as np
import torch
from tqdm import tqdm

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from algorithms.agent import ConstrainedPPOAgent

from utils.buffers import Memory
from utils.colreg_handler import COLREGHandler

from colreg_logic import rtamt_yml_parser

model_name = "boat_agent_model"
# None - use the Unity Editor (press Play)
unity_env_path = None

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = "cpu"
# BoatAgent Parameters - must match those in Unity
OBSERVATION_SIZE = 24 # From UnityEnvironment/Scripts/BoatAgent.cs
RAYCAST_COUNT = 7 # 3 side rays + 1 front ray # From Unity RayPerceptionSensorComponent3D
RAYCAST_SIZE = RAYCAST_COUNT * 2 # Each ray (7) has a distance and a hit flag (1 or 0)

INPUT_SIZE = OBSERVATION_SIZE + RAYCAST_SIZE

ACTION_SIZE = 2 # Left Jet, Right Jet
BEHAVIOR_NAME = "BoatAgent"

testing = True
EPISODES = 10
ROLLOUT_SIZE = 2_048
TOT_STEPS = 3_000

SAVE_INTERVAL = 10_000

colreg_path = "colreg_logic/colreg.yaml"

loading_step = 2_048 # Set to a specific step number to load that checkpoint, or None to start fresh

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

    starting_step = 0

    colreg_handler = COLREGHandler()

    RTAMT = rtamt_yml_parser.RTAMTYmlParser(colreg_path)

    # Channel used to speed up the game time
    engine_config = EngineConfigurationChannel()
    
    print("Loading environment...")
    env = UnityEnvironment(file_name=unity_env_path, side_channels=[engine_config])
    
    env.reset()
    print("Environment loaded successfully.")
    
    # time_scale = 1.0 real tiem - 20.0 is 20x faster than real time
    engine_config.set_configuration_parameters(width=800, height=600, time_scale=10.0)

    # Debug info print behaviors available
    print("Behaviors found:", list(env.behavior_specs.keys()))
    behavior_name = list(env.behavior_specs.keys())[0] 
    
    agent = ConstrainedPPOAgent(INPUT_SIZE, ACTION_SIZE, device=DEVICE)

    if loading_step is not None:
        checkpoint_path = f"Models/{model_name}_{loading_step}.pth"
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
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
        
        # Progress bar per il training
        pbar = tqdm(total=TOT_STEPS, desc="Training", unit="steps")

        save_model = False
        
        while s < TOT_STEPS: 
            
            while (len(memory_buffer.states) < ROLLOUT_SIZE):
                
                obs, vec_obs = get_single_agent_obs(decision_steps)
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)

                action_tensor, log_probabs = agent.get_action(obs_tensor)
                action_numpy = action_tensor.detach().cpu().numpy()
                action_tuple = ActionTuple()
                action_tuple.add_continuous(action_numpy)

                env.set_actions(behavior_name, action_tuple)
                env.step()
                s += 1
                pbar.update(1)

                decision_steps, terminal_steps = env.get_steps(behavior_name)
                end_episode = len(terminal_steps) > 0

                if end_episode:
                    next_obs = terminal_steps.obs
                    reward = terminal_steps.reward
                else:
                    next_obs = decision_steps.obs
                    reward = decision_steps.reward

                memory_buffer.add_ppo_transition(
                state=obs_tensor, 
                action=action_tensor, 
                logprob=log_probabs,
                reward=reward, 
                is_terminal=float(end_episode)
                )

                
                r1 = colreg_handler.get_R1_robustness(obs=vec_obs)

                physical_speed = colreg_handler.get_ego_speed(vec_obs)

                memory_buffer.add_stl_sample(phys_speed=float(physical_speed), r1_robustness=float(r1))
                
                if memory_buffer.is_stl_ready():
                    #print("STL ready")
                    _ , single_rho = RTAMT.compute_robustness_dense(memory_buffer.stl_window)
                    
                    rho_1 = single_rho.get('R1_safe_distance', -10.0)
                    rho_2 = single_rho.get('R2_safe_speed', 0.0)
                   
                    cost_1 = max(0, -rho_1)
                    cost_2 = max(0, -rho_2)

                    memory_buffer.add_robustness(r1=rho_1,r2=rho_2)
                    memory_buffer.add_costs(c_r1=cost_1, c_r2=cost_2)
                else: 
                    memory_buffer.add_robustness(r1=colreg_handler.MAX_ROBUSTNESS_CAP,r2=colreg_handler.MAX_ROBUSTNESS_CAP)
                    memory_buffer.add_costs(c_r1=0.0, c_r2=0.0)

                if end_episode:
                    memory_buffer.clear_stl_window()
                    env.reset()
                    decision_steps, terminal_steps = env.get_steps(behavior_name)

                if s % SAVE_INTERVAL == 0:
                    save_model = True


            next_state = get_single_agent_obs(decision_steps)[0]

            rollout_buffer = {}
            rollout_buffer['states'] =  memory_buffer.states
            rollout_buffer['actions'] = memory_buffer.actions
            rollout_buffer['logprobs'] = memory_buffer.logprobs
            rollout_buffer['rewards'] = np.array(memory_buffer.rewards)
            rollout_buffer['masks'] = 1 - np.array(memory_buffer.is_terminals)
            rollout_buffer['next_state'] = np.array(next_state)
            rollout_buffer['cost_r1'] = np.array(memory_buffer.cost_r1)
            rollout_buffer['cost_r2'] = np.array(memory_buffer.cost_r2)

            robustness_dict = {'R1': min(memory_buffer.robustness_1), 'R2': min(memory_buffer.robustness_2)}
            
            log_dict = agent.update(rollouts=rollout_buffer,robustness_dict=robustness_dict,current_step=s)

            reward_returns = log_dict['reward'][1]

            memory_buffer.clear_ppo()

            pbar.write(f"Update, reward: {reward_returns.mean():.6f}, robustness R1: {robustness_dict['R1']:.6f}, robustness R2: {robustness_dict['R2']:.6f}")

            pbar.set_postfix({
                'Reward': f"{reward_returns.mean():.2f}",
                'R1': f"{robustness_dict['R1']:.2f}",
                'R2': f"{robustness_dict['R2']:.2f}"
            })

            # Save the model occasionally
            if save_model:
                print(f"Saving model at step {s}...")
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
                torch.save(checkpoint, f"Models/{model_name}_{s}.pth")
                save_model = False

    except KeyboardInterrupt:
        print("Manual interruption...")

    finally:
        pbar.close()
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()