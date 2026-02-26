import torch
import torch.optim as optim
import numpy as np
from algorithms.networks import Policy, Value, CostValue
from utils.cagrad import Cagrad_all
from copy import deepcopy

class ConstrainedPPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, ppo_eps=0.2, start_safety=40_960, device='cpu', 
                 entropy_coeff=0.01):
        self.gamma = gamma
        self.ppo_eps = ppo_eps
        self.start_safety = start_safety
        self.device = device
        self.entropy_coeff = entropy_coeff
        self.policy_net = Policy(state_size, action_size).to(self.device)
        self.value_net = Value(state_size).to(self.device)        
        # COLREG Cost Networks
        # Rule R1: Vessels always have to keep a safe distance between each other
        self.cost_net_safe_distance = CostValue(state_size).to(self.device)

        # Rule R2: A vessel shall always maintain a safe speed 
        self.cost_net_safe_speed = CostValue(state_size).to(self.device)   
        
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

    def _generate_batches(self, total_size, batch_size):
        indices = torch.randperm(total_size)
        for start_idx in range(0, total_size, batch_size):
            yield indices[start_idx:start_idx + batch_size]

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
        entropy = dist.entropy().sum(dim=-1)
        return action_log_probs, entropy

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
        gae_returns = self._calculate_gae(rewards, value_preds, next_value_pred, masks)
        adv_reward = gae_returns - value_preds

        # Cost R1 (Minimization) 
        # Using "cumulative_cost" to indicate sum of future penalties
        r1_cumulative_cost = self._calculate_gae(cost_r1, r1_cost_preds, r1_next_cost_pred, masks)
        adv_r1 = r1_cumulative_cost - r1_cost_preds

        # Cost R2 (Minimization)
        r2_cumulative_cost = self._calculate_gae(cost_r2, r2_cost_preds, r2_next_cost_pred, masks)
        adv_r2 = r2_cumulative_cost - r2_cost_preds

        return {
            "reward": (adv_reward, gae_returns),
            "r1": (adv_r1, r1_cumulative_cost),
            "r2": (adv_r2, r2_cumulative_cost)
        }

    def update(self, rollouts, robustness_dict, current_step, entropy_coeff=None):
        """
        Args:
            rollouts (dict): Dictionary containing raw lists from the buffer:
                             ['states', 'actions', 'rewards', 'masks', 
                              'cost_r1', 'cost_r2', 'next_state', 'logprobs']
            robustness_dict (dict): Current min robustness values e.g. {'R1': -0.1, 'R2': 0.5}
        """
        if entropy_coeff is None:
            entropy_coeff = self.entropy_coeff
        
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
        adv_reward, gae_returns = advantages["reward"]
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
        value_loss = torch.nn.MSELoss()(value_preds, gae_returns)
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
        cur_log_probs, entropy = self.evaluate_actions(states, actions)
        ratio = torch.exp(cur_log_probs - old_log_probs)

        # Entropy regularization to encourage exploration, scaled by coefficient, negaive beacuse we want to maximize entropy and optimizers minimize loss
        entropy_loss = -entropy_coeff * entropy.mean()

        violated_rules = [rule for rule, rho in robustness_dict.items() if rho < 0]
        # Case 1: No violations (NOMINAL MODE) -> Standard PPO update using reward advantage
        if not violated_rules or current_step < self.start_safety:
            actual_mode = "NOMINAL"
            if violated_rules:
                actual_mode="NOMINAL (warmup)"
            policy_loss = self._get_ppo_loss(ratio, adv_reward) + entropy_loss
            self.policy_opt.zero_grad()
            policy_loss.backward()

        # Case 2: single violation -> Minimize specific cost (Maximize negative cost advantage)
        elif len(violated_rules) == 1:
            rule = violated_rules[0]
            actual_mode = f"SINGLE VIOLATION {rule}"
            cost_adv = cost_config[rule]["adv"]
            # -cost_adv because we want to minimize cost
            policy_loss = self._get_ppo_loss(ratio, -cost_adv) + entropy_loss
            self.policy_opt.zero_grad()
            policy_loss.backward()

        # Case 3: multiple violations -> Use CAGrad to find the best update direction
        else:
            actual_mode = f"MULTIPLE VIOLATIONS {violated_rules}"
            grads = []
            for rule in violated_rules:
                cost_adv = cost_config[rule]["adv"]
                policy_loss = self._get_ppo_loss(ratio, -cost_adv) 
                #is_last = (rule == violated_rules[-1])
                flat_grad = self._compute_flat_grad(policy_loss, self.policy_net, retain_graph=True)
                if flat_grad is not None:
                    grads.append(flat_grad)
            
            if grads:
                grad_vec = torch.stack(grads)
                merged_grad = self.cagrad_helper.cagrad(grad_vec, num_tasks=len(violated_rules))
                entropy_grad = self._compute_flat_grad(entropy_loss, self.policy_net, retain_graph=False)
                if entropy_grad is not None:
                    merged_grad += entropy_grad
                self._set_flat_grad(merged_grad, self.policy_net)
        
        # clip gradient to prevent exploding gradients (optional, can be tuned or removed)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)

        self.policy_opt.step()

        return {
            "mode": actual_mode,
            "violated_rules": violated_rules,
            "robustness": {rule: robustness_dict[rule] for rule in violated_rules},
            "reward": (adv_reward, gae_returns),
            "r1": (adv_r1, r1_cumulative_cost),
            "r2": (adv_r2, r2_cumulative_cost)
        }
    
    # We iterate n_epochs times over the collected data to increase sample efficiency.
    # In each epoch, we shuffle the data and perform updates on small random subsets (minibatches) of dimension batch_size.
    def update_with_minibatches(self, rollouts, robustness_dict, current_step, entropy_coeff=None, n_epochs=10, batch_size=64):
        """
        Args:
            rollouts (dict): Dictionary containing raw lists from the buffer:
                             ['states', 'actions', 'rewards', 'masks', 
                              'cost_r1', 'cost_r2', 'next_state', 'logprobs']
            robustness_dict (dict): Current min robustness values e.g. {'R1': -0.1, 'R2': 0.5}
            entropy_coeff (float): Coefficient for entropy regularization
            n_epochs (int): Number of times to iterate over the entire buffer
            batch_size (int): Size of the mini-batches
        """
        if entropy_coeff is None:
            entropy_coeff = self.entropy_coeff
        
        # Calculate Advantages for reward and costs (ON FULL BUFFER to preserve temporal dependencies)
        states = torch.stack(rollouts['states']).to(self.device).detach()
        actions = torch.stack(rollouts['actions']).to(self.device).detach()
        rewards = torch.from_numpy(rollouts['rewards']).float().to(self.device).detach().squeeze()
        old_log_probs = torch.stack(rollouts['logprobs']).to(self.device).detach().squeeze()
        masks = torch.from_numpy(rollouts['masks']).float().to(self.device).detach()
        cost_r1 = torch.from_numpy(rollouts['cost_r1']).float().to(self.device).detach()
        cost_r2 = torch.from_numpy(rollouts['cost_r2']).float().to(self.device).detach()
        next_state = torch.from_numpy(rollouts['next_state']).float().unsqueeze(0).to(self.device).detach()

        advantages = self.compute_all_advantages(states, next_state, rewards, cost_r1, cost_r2, masks)
        adv_reward, gae_returns = advantages["reward"]
        adv_r1, r1_cumulative_cost = advantages["r1"]
        adv_r2, r2_cumulative_cost = advantages["r2"]
        
        # Normalize advantages to stabilize training (on the whole buffer)
        #adv_reward = (adv_reward - adv_reward.mean()) / (adv_reward.std() + 1e-8)

        # not normalizing cost advantage to preserve scale for cagrad
        #adv_r1 = (adv_r1 - adv_r1.mean()) / (adv_r1.std() + 1e-8)
        #adv_r2 = (adv_r2 - adv_r2.mean()) / (adv_r2.std() + 1e-8)

        # Determine the training mode once for the entire update
        violated_rules = [rule for rule, rho in robustness_dict.items() if rho < 0]
        # Case 1: No violations (NOMINAL MODE)
        if not violated_rules or current_step < self.start_safety:
            actual_mode = "NOMINAL"
            if violated_rules:
                actual_mode="NOMINAL (warmup)"
        # Case 2: single violation 
        elif len(violated_rules) == 1:
            actual_mode = f"SINGLE VIOLATION {violated_rules[0]}"
        # Case 3: multiple violations
        else:
            actual_mode = f"MULTIPLE VIOLATIONS {violated_rules}"

        pg_losses, v_losses, ent_vals = [], [], []
        c_losses_r1, c_losses_r2 = [], []

        total_samples = states.size(0)

        # PPO Multi-Epoch & Minibatch Training Loop
        for epoch in range(n_epochs):
            for batch_indices in self._generate_batches(total_samples, batch_size):
                
                # Extract mini-batch tensors
                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_gae_returns = gae_returns[batch_indices]
                b_adv_reward = adv_reward[batch_indices]
                b_adv_r1 = adv_r1[batch_indices]
                b_r1_cum_cost = r1_cumulative_cost[batch_indices]
                b_adv_r2 = adv_r2[batch_indices]
                b_r2_cum_cost = r2_cumulative_cost[batch_indices]

                # Normalize advantage on mini-batch for reward as done in stable-baseline3
                b_adv_reward = (b_adv_reward - b_adv_reward.mean()) / (b_adv_reward.std() + 1e-8)

                # setup for update (batched version)
                b_cost_config = {
                    "R1": {
                        "adv": b_adv_r1,
                        "cumulative_cost": b_r1_cum_cost,
                        "network": self.cost_net_safe_distance,
                        "optimizer": self.cost_opts[0]
                    },
                    "R2": {
                        "adv": b_adv_r2,
                        "cumulative_cost": b_r2_cum_cost,
                        "network": self.cost_net_safe_speed,
                        "optimizer": self.cost_opts[1]
                    }
                }

                # update Value critic
                self.value_opt.zero_grad()
                value_preds = self.value_net(b_states).squeeze()
                value_loss = torch.nn.MSELoss()(value_preds, b_gae_returns)
                value_loss.backward()
                self.value_opt.step()
                v_losses.append(value_loss.item())

                # update Cost critics
                for rule, config in b_cost_config.items():
                    net = config["network"]
                    opt = config["optimizer"]
                    cumulative_cost = config["cumulative_cost"]

                    opt.zero_grad()
                    cost_preds = net(b_states).squeeze()
                    cost_loss = torch.nn.MSELoss()(cost_preds, cumulative_cost)
                    cost_loss.backward()
                    opt.step()
                    if rule == "R1": c_losses_r1.append(cost_loss.item())
                    if rule == "R2": c_losses_r2.append(cost_loss.item())
                
                # Policy Update Logic
                cur_log_probs, entropy = self.evaluate_actions(b_states, b_actions)
                # clamp to prevent nan crashes
                ratio = torch.exp(torch.clamp(cur_log_probs - b_old_log_probs, min=-20.0, max=5.0))

                # Entropy regularization to encourage exploration, scaled by coefficient, negaive beacuse we want to maximize entropy and optimizers minimize loss
                entropy_loss = -entropy_coeff * entropy.mean()
                ent_vals.append(entropy.mean().item())

                # Case 1: No violations (NOMINAL MODE) -> Standard PPO update using reward advantage
                if not violated_rules or current_step < self.start_safety:
                    policy_loss = self._get_ppo_loss(ratio, b_adv_reward) + entropy_loss
                    self.policy_opt.zero_grad()
                    policy_loss.backward()
                    pg_losses.append(policy_loss.item())

                # Case 2: single violation -> Minimize specific cost (Maximize negative cost advantage)
                elif len(violated_rules) == 1:
                    rule = violated_rules[0]
                    cost_adv = b_cost_config[rule]["adv"]
                    # -cost_adv because we want to minimize cost
                    policy_loss = self._get_ppo_loss(ratio, -cost_adv) + entropy_loss
                    self.policy_opt.zero_grad()
                    policy_loss.backward()
                    pg_losses.append(policy_loss.item())

                # Case 3: multiple violations -> Use CAGrad to find the best update direction
                else:
                    batch_pg_losses = []
                    grads = []
                    for rule in violated_rules:
                        cost_adv = b_cost_config[rule]["adv"]
                        policy_loss = self._get_ppo_loss(ratio, -cost_adv)
                        batch_pg_losses.append(policy_loss.item()) 
                        flat_grad = self._compute_flat_grad(policy_loss, self.policy_net, retain_graph=True)
                        if flat_grad is not None:
                            grads.append(flat_grad)
                    
                    if grads:
                        grad_vec = torch.stack(grads)
                        merged_grad = self.cagrad_helper.cagrad(grad_vec, num_tasks=len(violated_rules))
                        entropy_grad = self._compute_flat_grad(entropy_loss, self.policy_net, retain_graph=False)
                        if entropy_grad is not None:
                            merged_grad += entropy_grad
                        self._set_flat_grad(merged_grad, self.policy_net)
                        pg_losses.append(np.mean(batch_pg_losses))
                
                # clip gradient to prevent exploding gradients (optional, can be tuned or removed)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)

                self.policy_opt.step()

        # Return full tensors for accurate logging in train.py
        return {
            "mode": actual_mode,
            "violated_rules": violated_rules,
            "robustness": {rule: robustness_dict[rule] for rule in violated_rules},
            "reward": (adv_reward, gae_returns),
            "r1": (adv_r1, r1_cumulative_cost),
            "r2": (adv_r2, r2_cumulative_cost),
            "policy_loss": np.mean(pg_losses) if pg_losses else 0,
            "value_loss": np.mean(v_losses) if v_losses else 0,
            "entropy": np.mean(ent_vals) if ent_vals else 0,
            "cost_loss_r1": np.mean(c_losses_r1) if c_losses_r1 else 0,
            "cost_loss_r2": np.mean(c_losses_r2) if c_losses_r2 else 0
        }