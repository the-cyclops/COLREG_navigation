import torch
import torch.optim as optim
import numpy as np
from algorithms.agent import ConstrainedPPOAgent
from algorithms.networks import Policy, Value, CostValue
from utils.cagrad import Cagrad_all

class RewardShapingPPOAgent(ConstrainedPPOAgent):
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, ppo_eps=0.2, device='cpu', 
                 entropy_coeff=0.0, penalty_scale=1.0):

        # __init__.super() is not used to avoid instantiating the cost networks
        
        self.gamma = gamma
        self.ppo_eps = ppo_eps
        self.device = device
        self.entropy_coeff = entropy_coeff
        self.penalty_scale = penalty_scale
        self.start_safety = 0 # Compatibility Dummy

        self.policy_net = Policy(state_size, action_size).to(self.device)
        self.value_net = Value(state_size).to(self.device)        
        
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=lr)

        self.total_early_stops = 0

    def set_train_mode(self):
        self.policy_net.train()
        self.value_net.train()

    def set_eval_mode(self):
        self.policy_net.eval()
        self.value_net.eval()

    def compute_advantages_shaped(self, states, next_state, shaped_rewards, masks):
        with torch.no_grad():
            value_preds = self.value_net(states).squeeze()
            next_value_pred = self.value_net(next_state).item()

        # Ereditato dalla classe madre!
        gae_returns = self._calculate_gae(shaped_rewards, value_preds, next_value_pred, masks)
        adv_reward = gae_returns - value_preds
        return adv_reward, gae_returns

    def update(self, rollouts, robustness_dict, current_step, entropy_coeff=None, n_epochs=10, batch_size=64, target_kl=None, writer=None):
        if entropy_coeff is None:
            entropy_coeff = self.entropy_coeff
        
        states = torch.stack(rollouts['states']).to(self.device).detach().squeeze()
        actions = torch.stack(rollouts['actions']).to(self.device).detach().squeeze()
        rewards = torch.from_numpy(rollouts['rewards']).float().to(self.device).detach().squeeze()
        old_log_probs = torch.stack(rollouts['logprobs']).to(self.device).detach().squeeze()
        masks = torch.from_numpy(rollouts['masks']).float().to(self.device).detach()
        
        cost_r1 = torch.from_numpy(rollouts['cost_r1']).float().to(self.device).detach()
        cost_r2 = torch.from_numpy(rollouts['cost_r2']).float().to(self.device).detach()
        cost_r6 = torch.from_numpy(rollouts['cost_r6']).float().to(self.device).detach()
        next_state = torch.from_numpy(rollouts['next_state']).float().unsqueeze(0).to(self.device).detach()     

        # ========================================================
        # REWARD SHAPING (Applichiamo la penalità solo se > 0)
        # ========================================================
        total_penalty = torch.relu(cost_r1) + torch.relu(cost_r2) + torch.relu(cost_r6)
        shaped_rewards = rewards - (self.penalty_scale * total_penalty)

        # Vantaggi calcolati sul reward fuso
        adv_reward, gae_returns = self.compute_advantages_shaped(states, next_state, shaped_rewards, masks)
        
        pg_losses, v_losses, ent_vals = [], [], []
        total_samples = states.size(0)
        
        for epoch in range(n_epochs):
            batch_seed = current_step + epoch
            # _generate_batches è ereditato dalla classe madre!
            for batch_indices in self._generate_batches(total_samples, batch_size, seed=batch_seed):
                
                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_gae_returns = gae_returns[batch_indices]
                b_adv_reward = adv_reward[batch_indices]

                b_adv_reward = (b_adv_reward - b_adv_reward.mean()) / (b_adv_reward.std() + 1e-8)

                # 1. Update Value Network
                self.value_opt.zero_grad()
                value_preds = self.value_net(b_states).squeeze()
                value_loss = torch.nn.MSELoss()(value_preds, b_gae_returns)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                self.value_opt.step()
                v_losses.append(value_loss.item())

                # 2. Update Policy Network
                cur_log_probs, entropy = self.evaluate_actions(b_states, b_actions)
                ratio = torch.exp(cur_log_probs - b_old_log_probs)
                
                entropy_loss = -entropy_coeff * entropy.mean()
                
                policy_loss = self._get_ppo_loss(ratio, b_adv_reward) + entropy_loss
                
                self.policy_opt.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                self.policy_opt.step()
                
                pg_losses.append(policy_loss.item())
                ent_vals.append(entropy.mean().item())

        return {
            "mode": "REWARD SHAPING BASELINE",
            "violated_rules": [],
            "reward": (adv_reward, gae_returns),
            "policy_loss": np.mean(pg_losses) if pg_losses else 0,
            "value_loss": np.mean(v_losses) if v_losses else 0,
            "entropy": np.mean(ent_vals) if ent_vals else 0,
        }