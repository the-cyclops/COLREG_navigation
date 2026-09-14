from algorithms.agent import ConstrainedPPOAgent
import numpy as np
import torch
import random

class RandomCMORLAgent(ConstrainedPPOAgent):
    def __init__(self, *args,seed=42, **kwargs):
        super().__init__(*args, **kwargs)

        self.seed = seed
        self.rng = random.Random(seed)

        if hasattr(self, "cagrad_helper"):
            del self.cagrad_helper

    def update(self, rollouts, robustness_dict, current_step, entropy_coeff=None, n_epochs=10, batch_size=64, target_kl=None, writer=None):
        if entropy_coeff is None:
            entropy_coeff = self.entropy_coeff
        
        # 1. Caricamento tensori dal buffer
        states = torch.stack(rollouts['states']).to(self.device).detach().squeeze()
        actions = torch.stack(rollouts['actions']).to(self.device).detach().squeeze()
        rewards = torch.from_numpy(rollouts['rewards']).float().to(self.device).detach().squeeze()
        old_log_probs = torch.stack(rollouts['logprobs']).to(self.device).detach().squeeze()
        masks = torch.from_numpy(rollouts['masks']).float().to(self.device).detach()
        cost_r1 = torch.from_numpy(rollouts['cost_r1']).float().to(self.device).detach()
        cost_r2 = torch.from_numpy(rollouts['cost_r2']).float().to(self.device).detach()
        cost_r6 = torch.from_numpy(rollouts['cost_r6']).float().to(self.device).detach()
        next_state = torch.from_numpy(rollouts['next_state']).float().unsqueeze(0).to(self.device).detach()     

        # 2. Calcolo advantages
        advantages = self.compute_all_advantages(states, next_state, rewards, cost_r1, cost_r2, cost_r6, masks)
        adv_reward, gae_returns = advantages["reward"]
        adv_r1, r1_cumulative_cost = advantages["r1"]
        adv_r2, r2_cumulative_cost = advantages["r2"]
        adv_r6, r6_cumulative_cost = advantages["r6"]

        # 3. Identificazione violazioni e selezione vincolo casuale
        violated_rules = sorted([rule for rule, rho in robustness_dict.items() if rho < 0])
        
        chosen_rule = None
        if not violated_rules or current_step < self.start_safety:
            actual_mode = "NOMINAL"
            if violated_rules:
                actual_mode = "NOMINAL (warmup)"
        else:
            # Baseline casuale: estrazione uniforme tra i vincoli violati
            chosen_rule = self.rng.choice(violated_rules)
            if len(violated_rules) == 1:
                actual_mode = f"SINGLE VIOLATION {chosen_rule}"
            else:
                actual_mode = f"MULTIPLE VIOLATIONS (random constraint: {chosen_rule} out of {violated_rules})"

        # 4. Normalizzazione standard degli advantage dei costi
        adv_r1 = (adv_r1 - adv_r1.mean()) / (adv_r1.std() + 1e-8)
        adv_r2 = (adv_r2 - adv_r2.mean()) / (adv_r2.std() + 1e-8)
        adv_r6 = (adv_r6 - adv_r6.mean()) / (adv_r6.std() + 1e-8)

        if writer is not None:
            writer.add_scalar("Debug_Adv/Reward_Mean", adv_reward.mean().item(), current_step)
            writer.add_scalar("Debug_Adv/Reward_Std", adv_reward.std().item(), current_step)
            writer.add_scalar("Debug_Adv/R1_Mean", adv_r1.mean().item(), current_step)
            writer.add_scalar("Debug_Adv/R2_Mean", adv_r2.mean().item(), current_step)
            writer.add_scalar("Debug_Adv/R6_Mean", adv_r6.mean().item(), current_step)

        pg_losses, v_losses, ent_vals = [], [], []
        c_losses_r1, c_losses_r2, c_losses_r6 = [], [], []
        policy_grad_norms, value_grad_norms = [], []
        r1_grad_norms, r2_grad_norms, r6_grad_norms = [], [], []
        kl_divs, clip_fractions = [], []

        total_samples = states.size(0)
        continue_training = True

        # 5. Training Loop PPO
        for epoch in range(n_epochs):
            batch_seed = current_step + epoch
            for batch_indices in self._generate_batches(total_samples, batch_size, seed=batch_seed):
                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_gae_returns = gae_returns[batch_indices]
                b_adv_reward = adv_reward[batch_indices]
                b_adv_r1 = adv_r1[batch_indices]
                b_r1_cum_cost = r1_cumulative_cost[batch_indices]
                b_adv_r2 = adv_r2[batch_indices]
                b_r2_cum_cost = r2_cumulative_cost[batch_indices]
                b_adv_r6 = adv_r6[batch_indices]
                b_r6_cum_cost = r6_cumulative_cost[batch_indices]

                b_adv_reward = (b_adv_reward - b_adv_reward.mean()) / (b_adv_reward.std() + 1e-8)

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
                    },
                    "R6": {
                        "adv": b_adv_r6,
                        "cumulative_cost": b_r6_cum_cost,
                        "network": self.cost_net_R6,
                        "optimizer": self.cost_opts[2]
                    }
                }

                # Aggiornamento Value critic (Reward)
                self.value_opt.zero_grad()
                value_preds = self.value_net(b_states).squeeze()
                value_loss = torch.nn.MSELoss()(value_preds, b_gae_returns)
                value_loss.backward()
                val_norm = torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                value_grad_norms.append(val_norm.item())
                self.value_opt.step()
                v_losses.append(value_loss.item())

                # Aggiornamento Cost critics (tutti i critici continuano ad allenarsi)
                for rule, config in b_cost_config.items():
                    net, opt = config["network"], config["optimizer"]
                    opt.zero_grad()
                    cost_preds = net(b_states).squeeze()
                    cost_loss = torch.nn.MSELoss()(cost_preds, config["cumulative_cost"])
                    cost_loss.backward()
                    g_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                    opt.step()

                    if rule == "R1":
                        c_losses_r1.append(cost_loss.item())
                        r1_grad_norms.append(g_norm.item())
                    elif rule == "R2":
                        c_losses_r2.append(cost_loss.item())
                        r2_grad_norms.append(g_norm.item())
                    elif rule == "R6":
                        c_losses_r6.append(cost_loss.item())
                        r6_grad_norms.append(g_norm.item())

                # Aggiornamento Policy
                cur_log_probs, entropy = self.evaluate_actions(b_states, b_actions)
                ratio = torch.exp(cur_log_probs - b_old_log_probs)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    kl_divs.append(approx_kl)
                    is_clipped = ((ratio < (1 - self.ppo_eps)) | (ratio > (1 + self.ppo_eps))).float().mean().item()
                    clip_fractions.append(is_clipped)

                if target_kl is not None and approx_kl > 1.5 * target_kl:
                    self.total_early_stops += 1
                    continue_training = False
                    break

                entropy_loss = -entropy_coeff * entropy.mean()
                ent_vals.append(entropy.mean().item())

                # Calcolo loss della policy:
                # - Se non ci sono violazioni -> ottimizza Reward
                # - Se ci sono violazioni -> ottimizza il vincolo casuale selezionato
                if chosen_rule is None:
                    policy_loss = self._get_ppo_loss(ratio, b_adv_reward) + entropy_loss
                else:
                    target_cost_adv = b_cost_config[chosen_rule]["adv"]
                    policy_loss = self._get_ppo_loss(ratio, -target_cost_adv) + entropy_loss

                self.policy_opt.zero_grad()
                policy_loss.backward()
                p_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                policy_grad_norms.append(p_norm.item())
                self.policy_opt.step()
                pg_losses.append(policy_loss.item())

            if not continue_training:
                break

        if writer is not None:
            writer.add_scalar("Debug_Policy/Avg_Grad_Norm", np.mean(policy_grad_norms), current_step)
            writer.add_scalar("Debug_Policy/Approx_KL", np.mean(kl_divs), current_step)
            writer.add_scalar("Debug_Policy/Clip_Fraction", np.mean(clip_fractions), current_step)
            writer.add_scalar("Debug_Policy/Total_Early_Stops", self.total_early_stops, current_step)
            writer.add_scalar("Debug_Grad_Norm/Value_Network", np.mean(value_grad_norms), current_step)
            writer.add_scalar("Debug_Grad_Norm/Cost_Network_R1", np.mean(r1_grad_norms), current_step)
            writer.add_scalar("Debug_Grad_Norm/Cost_Network_R2", np.mean(r2_grad_norms), current_step)
            writer.add_scalar("Debug_Grad_Norm/Cost_Network_R6", np.mean(r6_grad_norms), current_step)

        return {
            "mode": actual_mode,
            "violated_rules": violated_rules,
            "chosen_rule": chosen_rule,
            "robustness": {rule: robustness_dict[rule] for rule in violated_rules},
            "reward": (adv_reward, gae_returns),
            "r1": (adv_r1, r1_cumulative_cost),
            "r2": (adv_r2, r2_cumulative_cost),
            "r6": (adv_r6, r6_cumulative_cost),
            "policy_loss": np.mean(pg_losses) if pg_losses else 0,
            "value_loss": np.mean(v_losses) if v_losses else 0,
            "entropy": np.mean(ent_vals) if ent_vals else 0,
            "cost_loss_r1": np.mean(c_losses_r1) if c_losses_r1 else 0,
            "cost_loss_r2": np.mean(c_losses_r2) if c_losses_r2 else 0,
            "cost_loss_r6": np.mean(c_losses_r6) if c_losses_r6 else 0,
        }