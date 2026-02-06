import numpy as np
import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from algorithms.agent import ConstrainedPPOAgent

model_name = "boat_agent_model"
# None - use the Unity Editor (press Play)
unity_env_path = None
models_path = "Models"
                       
# BoatAgent Parameters - must match those in Unity
OBSERVATION_SIZE = 24 # From UnityEnvironment/Scripts/BoatAgent.cs
RAYCAST_COUNT = 7 # 3 side rays + 1 front ray # From Unity RayPerceptionSensorComponent3D

INPUT_SIZE = OBSERVATION_SIZE + (RAYCAST_COUNT * 2)

ACTION_SIZE = 2 # Left Jet, Right Jet
BEHAVIOR_NAME = "BoatAgent"

testing = True
EPISODES = 10

class Network(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, output_dim)
        self.tanh = torch.nn.Tanh() # Importante per output continui tra -1 e 1

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.tanh(self.fc2(x))


def main():

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

    try:
        total_episodes = EPISODES
        
        for episode in range(total_episodes):
            env.reset()
            
            # Ottiene i primi dati dagli agenti
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # Tracking reward episodio
            episode_rewards = 0
            
            while len(decision_steps) > 0:
                # Decision_steps contain a list of observations from different sources
                # [0] is the ray_cast_perception sensor, [1] is the manual vector observation 

                obs = np.concatenate((decision_steps.obs[0], decision_steps.obs[1]), axis=1)
            
                obs_tensor = torch.from_numpy(obs).float()

                if testing:
                    print(f"Obs shape: {obs_tensor.shape}, Obs: {obs_tensor}")

                action_tensor = agent(obs_tensor)
                
                # Convert the action tensor to numpy and prepare ActionTuple for ML-Agents
                action_numpy = action_tensor.detach().numpy()
                action_tuple = ActionTuple()
                action_tuple.add_continuous(action_numpy)
                
                # Send the action to the environment
                env.set_actions(behavior_name, action_tuple)
                
                # physical step in the environment and advance of number of frames (default 5) specified in DecisionRequester 
                env.step()
                
                # decision_steps = Agents that need to make a decision (alive) - they receive obs and reward
                # terminal_steps = Agents that reached a terminal state (dead) - they receive obs and reward for the last step
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                
                #
                
                if testing:
                    optimizer.zero_grad()
                    loss = -torch.mean(action_tensor)  # Dummy loss for testing
                    loss.backward()
                    optimizer.step()
                else:
                    pass
                    #agent.update(decision_steps, episode_rewards)  
                
                # Example reading rewards for alive agents:
                if len(decision_steps) > 0:
                    episode_rewards += np.mean(decision_steps.reward)
                
                # Example reading rewards for dead agents (End of episode for them):    
                if len(terminal_steps) > 0:
                    episode_rewards += np.mean(terminal_steps.reward)

            print(f"Episode {episode} completed. Estimated average reward: {episode_rewards}")
            
            # Save the model occasionally
            if episode % 5 == 0:
                torch.save(agent.state_dict(), f"Models/{model_name}_{episode}.pth")

    except KeyboardInterrupt:
        print("Manual interruption...")

    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()