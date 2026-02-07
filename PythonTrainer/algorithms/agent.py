import torch
import torch.optim as optim
from algorithms.networks import Policy, Value
from utils.cagrad import Cagrad_all
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
        # CAGrad helper
        self.cagrad_helper = Cagrad_all(c=0.5)
    
    # ----- Helper Functions for Cagrad and GAE computation -----

    def _compute_flat_grad(self, loss, network, retain_graph=False):
        """
        Computes gradients for a specific loss and returns them as a flattened tensor.
        Does NOT perform an optimizer step.
        """
        # Clear previous gradients
        self.policy_opt.zero_grad()
        
        # Calculate gradients (retain_graph is needed if we reuse the graph for other losses)
        loss.backward(retain_graph=retain_graph)
        
        # Collect and flatten gradients from all parameters
        grads = []
        for p in network.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        
        # Concatenate into a single 1D tensor
        if not grads:
            return None
        return torch.cat(grads)

    def _set_flat_grad(self, flat_grad, network):
        """
        Manually sets the .grad attribute of the network parameters using a flattened tensor.
        This prepares the network for the optimizer step using the merged gradient.
        """
        
        idx = 0
        for p in network.parameters():
            if p.grad is not None:
                num_param = p.numel()
                # Slice the flat vector corresponding to this parameter
                grad_slice = flat_grad[idx : idx + num_param]
                
                # Reshape and assign to the .grad attribute
                p.grad = grad_slice.view_as(p)
                
                idx += num_param

    def _calculate_gae(self, rewards, values, next_value, masks, lam=0.95):
        """
        Generic GAE calculation. 
        Works for both Maximization (Reward) and Minimization (Cost) signals.
        masks: 0 if terminal state, 1 otherwise
        """
        gae = 0
        returns = []
        
        values = values.detach()
        
        # Iterate backwards through the trajectory
        for step in reversed(range(len(rewards))):
            # If it's the last step, use next_value (bootstrap), otherwise use next step's value
            next_val = next_value if step == len(rewards) - 1 else values[step + 1]
            
            # Delta: r + gamma * V(s') * mask - V(s)
            delta = rewards[step] + self.gamma * next_val * masks[step] - values[step]
            
            # GAE: delta + gamma * lambda * mask * previous_gae
            gae = delta + self.gamma * lam * masks[step] * gae
            
            # Return = Advantage + Value (Target for Value Network)
            returns.insert(0, gae + values[step])
            
        return torch.stack(returns)
    
    # ----- Main Functions -----

    def evaluate_actions(self, states, actions):
        action_mean, _, action_std = self.policy_net(states)
        dist = torch.distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(actions).sum(dim=-1)
        return action_log_probs

    def compute_all_advantages(self, states, next_state, rewards, cost_r1, cost_r2, masks):
        """
        Computes Advantages and Targets for Reward, Cost R1, and Cost R2.
        masks: 0 if terminal state, 1 otherwise
        """
        
        # 1. Get predictions (Estimates) for the whole batch
        with torch.no_grad():
            # Standard Reward Value (V)
            value_preds = self.value_net(states).squeeze()
            
            # Cost Estimates (V_cost)
            # Represents: Expected Discounted Cumulative Cost
            r1_cost_preds = self.cost_net_safe_distance(states).squeeze()
            r2_cost_preds = self.cost_net_safe_speed(states).squeeze()

            # 2. Get bootstrap predictions for next state
            next_value_pred = self.value_net(next_state).item()
            r1_next_cost_pred = self.cost_net_safe_distance(next_state).item()
            r2_next_cost_pred = self.cost_net_safe_speed(next_state).item()

        # 3. Main Reward (Maximization)
        reward_returns = self._calculate_gae(rewards, value_preds, next_value_pred, masks)
        adv_reward = reward_returns - value_preds

        # 4. Cost R1 (Minimization) 
        # Using "cumulative_cost" to indicate sum of future penalties
        r1_cumulative_cost = self._calculate_gae(cost_r1, r1_cost_preds, r1_next_cost_pred, masks)
        adv_r1 = r1_cumulative_cost - r1_cost_preds

        # 5. Cost R2 (Minimization)
        r2_cumulative_cost = self._calculate_gae(cost_r2, r2_cost_preds, r2_next_cost_pred, masks)
        adv_r2 = r2_cumulative_cost - r2_cost_preds

        return {
            "reward": (adv_reward, reward_returns),
            "r1": (adv_r1, r1_cumulative_cost),
            "r2": (adv_r2, r2_cumulative_cost)
        }

    def update(self, batch, robustness_values):
        """
        Main update loop implementing the Switching Controller logic.
        
        batch: Dict containing tensors for states, actions, returns, advantages.
        robustness_values: List or tensor of robustness values (negative = violation).
        """
        # 1. Calculate V-targets and Advantages for reward and costs
        stastes = batch["states"]
        # 2. If all robustness_values >= 0:
        #    -> Optimize Policy Loss (standard PPO) + Value Loss
        # 3. If one or more robustness_values < 0:
        #    -> Calculate gradients for each violated cost
        #    -> Use CAGrad to merge cost gradients
        #    -> Apply update to policy_net
        pass