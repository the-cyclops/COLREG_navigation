import numpy as np
import torch

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
models_path = "../Models"
                       
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
TOT_STEPS = 1_000_000
SAVE_INTERVAL = 1000

def main():

    colreg_handler = COLREGHandler()

    # Channel used to speed up the game time
    engine_config = EngineConfigurationChannel()
    
    print("Loading environment...")
    env = UnityEnvironment(file_name=unity_env_path, side_channels=[engine_config])
    
    env.reset()
    print("Environment loaded successfully.")
    
    # time_scale = 1.0 real tiem - 20.0 is 20x faster than real time
    engine_config.set_configuration_parameters(width=800, height=600, time_scale=20.0)

    # Debug info print behaviors available
    print("Behaviors found:", list(env.behavior_specs.keys()))
    behavior_name = list(env.behavior_specs.keys())[0] 
    
    if testing:
        # Instantiate the network and optimizer
        agent = Network(INPUT_SIZE, ACTION_SIZE)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    else:
        # Instantiate the PPO agent
        agent = ConstrainedPPOAgent(INPUT_SIZE, ACTION_SIZE)

    print(f"Start training on: {behavior_name}")

    memory_buffer = Memory()

    try:
        s = 0
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        while s < TOT_STEPS: 
            
            while (len(memory_buffer.states) < ROLLOUT_SIZE):
                # Decision_steps contain a list of observations from different sources
                # [0] is the ray_cast_perception sensor, [1] is the manual vector observation 
                if decision_steps.obs[0].shape[1] == RAYCAST_SIZE and decision_steps.obs[1].shape[1] == OBSERVATION_SIZE:
                    ray_obs = decision_steps.obs[0]
                    vec_obs = decision_steps.obs[1]
                elif decision_steps.obs[0].shape[1] == OBSERVATION_SIZE and decision_steps.obs[1].shape[1] == RAYCAST_SIZE:
                    ray_obs = decision_steps.obs[1]
                    vec_obs = decision_steps.obs[0]
                else:
                    raise ValueError("Unexpected observation shapes: ", decision_steps.obs[0].shape, decision_steps.obs[1].shape)
                
                obs = np.concatenate((ray_obs, vec_obs), axis=1)[0] # Get the first (and only) agent's observation

                obs_tensor = torch.from_numpy(obs).float()

                action_tensor, log_probabs = agent.get_action(obs_tensor)
                action_numpy = action_tensor.detach().numpy()
                action_tuple = ActionTuple()
                action_tuple.add_continuous(action_numpy)

                env.set_actions(behavior_name, action_tuple)
                env.step()
                s += 1

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

                # Retrieving Robustness using RTAMT and storing it in the buffer for later use in the PPO update
                r1 = colreg_handler.get_R1_robustness(obs=vec_obs)

                memory_buffer.add_robustness(r1=r1,r2=None)

                if end_episode:
                    env.reset()
                    decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            rollout_buffer = {}
            rollout_buffer['states'] = memory_buffer.states
            rollout_buffer['actions'] = memory_buffer.actions
            rollout_buffer['logprobs'] = memory_buffer.logprobs
            rollout_buffer['rewards'] = memory_buffer.rewards
            rollout_buffer['is_terminals'] = memory_buffer.is_terminals
            rollout_buffer['next_states'] = next_obs

            robustness_dict = {'R1': min(memory_buffer.robustness_1), 'R2': 0.5}

            agent.update(rollouts=rollout_buffer,robustness_dict=robustness_dict,current_step=s)
            
            # Save the model occasionally
            if s % SAVE_INTERVAL == 0:
                torch.save(agent.state_dict(), f"Models/{model_name}_{s}.pth")

    except KeyboardInterrupt:
        print("Manual interruption...")

    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()