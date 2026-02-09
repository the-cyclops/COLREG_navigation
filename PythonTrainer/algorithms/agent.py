import torch
import torch.optim as optim
from algorithms.networks import Policy, Value
from utils.cagrad import Cagrad_all
from copy import deepcopy

class ConstrainedPPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, ppo_eps=0.2, start_safety=40_960, device='cpu'):
        self.gamma = gamma
        self.ppo_eps = ppo_eps
        self.start_safety = start_safety
        self.device = device

        self.policy_net = Policy(state_size, action_size).to(self.device)
        self.value_net = Value(state_size).to(self.device)        
        # COLREG Cost Networks
        # Rule R1: Vessels always have to keep a safe distance between each other
        self.cost_net_safe_distance = Value(state_size).to(self.device)

        # Rule R2: A vessel shall always maintain a safe speed 
        self.cost_net_safe_speed = Value(state_size).to(self.device)   
        
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
            else:
                grads.append(torch.zeros_like(p).view(-1))

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
    
    def _get_ppo_loss(self, ratio, advantage):
        """
        Standard PPO clipped surrogate objective for reward maximization.
        To minimize cost, pass -cost_advantage.
        """
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.ppo_eps, 1 + self.ppo_eps) * advantage
        # return negative because we want to maximize the surrogate objective, but optimizers minimize loss
        return -torch.min(surr1, surr2).mean()
        
    
    # ----- Main Functions -----

    def get_action(self, state, deterministic=False):
        """
        Helper for training loop
        """
        with torch.no_grad():
            mean, _, std = self.policy_net(state)
            dist = torch.distributions.Normal(mean, std)
            
            if deterministic:
                action = mean
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action, log_prob

    def evaluate_actions(self, states, actions):
        action_mean, _, action_std = self.policy_net(states)
        dist = torch.distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(actions).sum(dim=-1)
        return action_log_probs

    def compute_all_advantages(self, states, next_state, rewards, cost_r1, cost_r2, masks):
        """
        Computes Advantages and Targets for Reward, Cost R1, and Cost R2.
        masks: 0 if terminal state, 1 otherwise

        returns: Dict with keys "reward", "r1", "r2", each containing a tuple of (advantages, returns/cumulative_costs)
        """
        
        # Get predictions (Estimates) for the whole batch
        with torch.no_grad():
            # Standard Reward Value (V)
            value_preds = self.value_net(states).squeeze()
            
            # Cost Estimates (V_cost)
            # Represents: Expected Discounted Cumulative Cost
            r1_cost_preds = self.cost_net_safe_distance(states).squeeze()
            r2_cost_preds = self.cost_net_safe_speed(states).squeeze()

            # Get bootstrap predictions for next state
            next_value_pred = self.value_net(next_state).item()
            r1_next_cost_pred = self.cost_net_safe_distance(next_state).item()
            r2_next_cost_pred = self.cost_net_safe_speed(next_state).item()

        # Main Reward (Maximization)
        reward_returns = self._calculate_gae(rewards, value_preds, next_value_pred, masks)
        adv_reward = reward_returns - value_preds

        # Cost R1 (Minimization) 
        # Using "cumulative_cost" to indicate sum of future penalties
        r1_cumulative_cost = self._calculate_gae(cost_r1, r1_cost_preds, r1_next_cost_pred, masks)
        adv_r1 = r1_cumulative_cost - r1_cost_preds

        # Cost R2 (Minimization)
        r2_cumulative_cost = self._calculate_gae(cost_r2, r2_cost_preds, r2_next_cost_pred, masks)
        adv_r2 = r2_cumulative_cost - r2_cost_preds

        return {
            "reward": (adv_reward, reward_returns),
            "r1": (adv_r1, r1_cumulative_cost),
            "r2": (adv_r2, r2_cumulative_cost)
        }

    def update(self, rollouts, robustness_dict, current_step):
        """
        Args:
            rollouts (dict): Dictionary containing raw lists from the buffer:
                             ['states', 'actions', 'rewards', 'masks', 
                              'cost_r1', 'cost_r2', 'next_state', 'logprobs']
            robustness_dict (dict): Current min robustness values e.g. {'R1': -0.1, 'R2': 0.5}
        """
        # Calculate Advantages for reward and costs
        states = torch.stack(rollouts['states']).to(self.device).detach()
        actions = torch.stack(rollouts['actions']).to(self.device).detach()
        rewards = torch.from_numpy(rollouts['rewards']).float().to(self.device).detach().squeeze()
        old_log_probs = torch.stack(rollouts['logprobs']).to(self.device).detach().squeeze()
        masks = torch.from_numpy(rollouts['masks']).float().to(self.device).detach()
        cost_r1 = torch.from_numpy(rollouts['cost_r1']).float().to(self.device).detach()
        cost_r2 = torch.from_numpy(rollouts['cost_r2']).float().to(self.device).detach()
        next_state = torch.from_numpy(rollouts['next_state']).float().unsqueeze(0).to(self.device).detach()

        advantages = self.compute_all_advantages(states, next_state, rewards, cost_r1, cost_r2, masks)
        adv_reward, reward_returns = advantages["reward"]
        adv_r1, r1_cumulative_cost = advantages["r1"]
        adv_r2, r2_cumulative_cost = advantages["r2"]
        # Normalize advantages to stabilize training
        adv_reward = (adv_reward - adv_reward.mean()) / (adv_reward.std() + 1e-8)

        # not normalizing cost advantage to preserve scale for cagrad
        #adv_r1 = (adv_r1 - adv_r1.mean()) / (adv_r1.std() + 1e-8)
        #adv_r2 = (adv_r2 - adv_r2.mean()) / (adv_r2.std() + 1e-8)

        # setup for update
        cost_config = {
            "R1": {
                "adv": adv_r1,
                "cumulative_cost": r1_cumulative_cost,
                "network": self.cost_net_safe_distance,
                "optimizer": self.cost_opts[0]
            },
            "R2": {
                "adv": adv_r2,
                "cumulative_cost": r2_cumulative_cost,
                "network": self.cost_net_safe_speed,
                "optimizer": self.cost_opts[1]
            }
        }

        # update Value critic
        self.value_opt.zero_grad()
        value_preds = self.value_net(states).squeeze()
        value_loss = torch.nn.MSELoss()(value_preds, reward_returns)
        value_loss.backward()
        self.value_opt.step()

        # update Cost critics
        for rule, config in cost_config.items():
            net = config["network"]
            opt = config["optimizer"]
            cumulative_cost = config["cumulative_cost"]

            opt.zero_grad()
            cost_preds = net(states).squeeze()
            cost_loss = torch.nn.MSELoss()(cost_preds, cumulative_cost)
            cost_loss.backward()
            opt.step()
        
        # Policy Update Logic
        cur_log_probs = self.evaluate_actions(states, actions)
        ratio = torch.exp(cur_log_probs - old_log_probs)

        violated_rules = [rule for rule, rho in robustness_dict.items() if rho < 0]
        # Case 1: No violations (NOMINAL MODE) -> Standard PPO update using reward advantage
        if not violated_rules or current_step < self.start_safety:
            actual_mode = "NOMINAL"
            if violated_rules:
                actual_mode="NOMINAL (warmup)"
            policy_loss = self._get_ppo_loss(ratio, adv_reward)
            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()
        # Case 2: single violation -> Minimize specific cost (Maximize negative cost advantage)
        elif len(violated_rules) == 1:
            rule = violated_rules[0]
            actual_mode = f"SINGLE VIOLATION {rule}"
            cost_adv = cost_config[rule]["adv"]
            # -cost_adv because we want to minimize cost
            policy_loss = self._get_ppo_loss(ratio, -cost_adv) 
            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()
        # Case 3: multiple violations -> Use CAGrad to find the best update direction
        else:
            actual_mode = f"MULTIPLE VIOLATIONS {violated_rules}"
            grads = []
            for rule in violated_rules:
                cost_adv = cost_config[rule]["adv"]
                policy_loss = self._get_ppo_loss(ratio, -cost_adv) 
                is_last = (rule == violated_rules[-1])
                flat_grad = self._compute_flat_grad(policy_loss, self.policy_net, retain_graph=not is_last)
                if flat_grad is not None:
                    grads.append(flat_grad)
            
            if grads:
                grad_vec = torch.stack(grads)
                merged_grad = self.cagrad_helper.cagrad(grad_vec, num_tasks=len(violated_rules))
                self._set_flat_grad(merged_grad, self.policy_net)
                self.policy_opt.step()
        return {
            "mode": actual_mode,
            "violated_rules": violated_rules,
            "robustness": {rule: robustness_dict[rule] for rule in violated_rules},
            "reward": (adv_reward, reward_returns),
            "r1": (adv_r1, r1_cumulative_cost),
            "r2": (adv_r2, r2_cumulative_cost)
        }