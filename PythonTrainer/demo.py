import torch
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from algorithms.agent import ConstrainedPPOAgent
from utils.buffers import Memory

# --- CONFIGURAZIONE DEMO ---
# Impostando None, Python aspetterà che tu prema "Play" in Unity
UNITY_ENV_PATH = None 
# Usa il seed dove hai ottenuto i risultati migliori
BEST_SEED = 34 
MODEL_PATH = f"Models/boat_agent_model_initial/seed_{BEST_SEED}/best_feasible_model.pth"

OBSERVATION_SIZE = 24 # From UnityEnvironment/Scripts/BoatAgent.cs
RAYCAST_COUNT = 7 # 3 side rays + 1 front ray # From Unity RayPerceptionSensorComponent3D
RAYCAST_SIZE = RAYCAST_COUNT * 2 # Each ray (7) has a distance and a hit flag (1 or 0)
NUM_ROBUSTNESS_FLAG = 2 # R1, R2

INPUT_SIZE = OBSERVATION_SIZE + RAYCAST_SIZE + NUM_ROBUSTNESS_FLAG
ACTION_SIZE = 2
DEVICE = "cpu"

def get_single_agent_obs(steps):
    raw_obs = steps.obs
    # In training.py controlli se raw_obs[0] è il Raycast (14) o la Vector Obs (24)
    if raw_obs[0].shape[1] == 14: 
        ray_obs = raw_obs[0][0]
        vec_obs = raw_obs[1][0]
    else:
        ray_obs = raw_obs[1][0]
        vec_obs = raw_obs[0][0]
    
    return np.concatenate((ray_obs, vec_obs)), vec_obs

def run_demo():
    # Inizializza l'agente e carica solo i pesi della Policy
    agent = ConstrainedPPOAgent(INPUT_SIZE, ACTION_SIZE, device=DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.policy_net.eval()

    engine_config = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=UNITY_ENV_PATH, side_channels=[engine_config])
    
    # IMPORTANTE: time_scale = 1.0 per vedere il movimento a velocità normale
    engine_config.set_configuration_parameters(time_scale=1.0)
    
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    memory = Memory(stl_horizon=20) # Horizon fittizio per i flag markoviani

    print("Connessione stabilita! Premi Play nell'Editor di Unity.")

    try:
        while True:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if len(terminal_steps) > 0:
                env.reset()
                memory.clear_stl_window()
                continue

            # Ottieni osservazioni e aggiungi i flag R1, R2 come nel training
            obs, _ = get_single_agent_obs(decision_steps)
            r1, r2 = memory.compute_markovian_flags()
            obs_augmented = np.concatenate((obs, [r1, r2]))
            
            obs_tensor = torch.from_numpy(obs_augmented).float().unsqueeze(0).to(DEVICE)

            # Azione deterministica (media) per la demo
            with torch.no_grad():
                action_tensor, _ = agent.get_action(obs_tensor, deterministic=False)
            
            action_numpy = action_tensor.cpu().numpy()
            print(f"Azioni inviate: {action_numpy}")
            action_tuple = ActionTuple()
            action_tuple.add_continuous(action_numpy)

            env.set_actions(behavior_name, action_tuple)
            env.step()
            
            # Aggiorna i flag (anche se non usiamo i costi, il modello si aspetta i flag corretti)
            memory.add_robustness(r1=0.5, r2=0.5) 

    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    run_demo()