import torch
import torch.optim as optim
from algorithms.networks import Policy, Value
from copy import deepcopy

class ConstrainedPPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        

        self.policy_net = Policy(state_size, action_size)
        self.value_net = Value(state_size)
        
        # COLREG Cost Networks
        # Rule R1: Vessels always have to keep a safe distance between each other
        self.cost_net_safe_distance = Value(state_size) 

        # Rule R2: A vessel shall always maintain a safe speed 
        self.cost_net_safe_speed = Value(state_size)
        
        # Optimizers
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=lr)
        self.cost_opts = [
            optim.Adam(self.cost_net_safe_distance.parameters(), lr=lr),
            optim.Adam(self.cost_net_safe_speed.parameters(), lr=lr)
        ]
    
    #def compute_gae

    def update(self, batch, robustness_values):
        """
        robustness_values: list of robustness values (RTAMT)
        If robustness < 0, the constraint is violated.
        """
        # 1. Calculate V-targets and Advantages for reward and costs
        # 2. If all robustness_values >= 0:
        #    -> Optimize Policy Loss (standard PPO) + Value Loss
        # 3. If one or more robustness_values < 0:
        #    -> Calculate gradients for each violated cost
        #    -> Use CAGrad to merge cost gradients
        #    -> Apply update to policy_net
        pass