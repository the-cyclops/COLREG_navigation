import torch
import torch.optim as optim
from algorithms.networks import Policy, Value
from copy import deepcopy

class ConstrainedPPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        
        # Reti (caricate da networks.py)
        self.policy_net = Policy(state_size, action_size)
        self.value_net = Value(state_size)
        
        # Definiamo i critici per i vincoli COLREG 
        # Rule R1: Vessels always have to keep a safe distance between each other
        self.cost_net_safe_distance = Value(state_size) 

        # Rule R2: A vessel shall always maintain a safe speed 
        self.cost_net_safe_speed = Value(state_size)
        
        # Ottimizzatori
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=lr)
        self.cost_opts = [
            optim.Adam(self.cost_net_safe_distance.parameters(), lr=lr),
            optim.Adam(self.cost_net_safe_speed.parameters(), lr=lr)
        ]

    def update(self, batch, robustness_values):
        """
        robustness_values: lista di valori di robustezza (RTAMT)
        Se robustness < 0, il vincolo è violato.
        """
        # 1. Calcola V-targets e Advantages per reward e costi
        # 2. Se tutti i robustness_values >= 0:
        #    -> Ottimizza Policy Loss (PPO standard) + Value Loss
        # 3. Se uno o più robustness_values < 0:
        #    -> Calcola i gradienti per ogni costo violato
        #    -> Usa CAGrad per fondere i gradienti dei costi
        #    -> Applica l'aggiornamento alla policy_net
        pass