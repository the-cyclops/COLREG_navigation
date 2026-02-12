from collections import deque

class Memory:
    def __init__(self, stl_horizon=40):
        # PPO buffers for on-policy training
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []
        
        # COLREG cost buffers populated via RTAMT robustness
        self.cost_r1 = []
        self.cost_r2 = []

        # Additional buffers for temporal logic evaluation
        self.robustness_1 = []
        self.robustness_2 = []

        # Sliding window for temporal logic evaluation (physical data)
        self.stl_window = deque(maxlen=stl_horizon)

        self.clear_stl_window()

    def add_ppo_transition(self, state, action, logprob, reward, is_terminal):
        """Store transition data for PPO update."""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def add_stl_sample(self, phys_speed, r1_signal):
        """Add denormalized physical data to the sliding window."""
        self.stl_window.append([phys_speed, r1_signal])

    def add_costs(self, c_r1, c_r2):
        """Store costs derived from RTAMT robustness values."""
        self.cost_r1.append(c_r1)
        self.cost_r2.append(c_r2)

    def add_robustness(self, r1, r2):
        """Store robustness values for later analysis."""
        self.robustness_1.append(r1)
        self.robustness_2.append(r2)

    def clear_ppo(self):
        """Clear on-policy buffers after policy update."""
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]
        del self.cost_r1[:]
        del self.cost_r2[:]
        del self.robustness_1[:]
        del self.robustness_2[:]

    def clear_stl_window(self):
        """Clear temporal window at the beginning of each episode."""
        self.stl_window.clear()
        # Initialize with safe defaults, 1.0 is the same as MAX_SAFETY_MARGIN_CAP in colreg_handler.py
        for _ in range(self.stl_window.maxlen):
            self.stl_window.append([0.0, 1.0]) 

    
    def compute_markovian_flags(self, R2_v_max=2.1):
        # R2_v_max is the same as V_max in the COLREG rules
        r1_rho = 0.0
        r2_rho = 0.0
        tau = self.stl_window.maxlen

        for sample in self.stl_window:
            # sample[0] = boat_speed, sample[1] = r1_signal

            #  R1 (distance): G[r1_signal >= 0]
            if sample[1] >= 0.0:
                r1_rho = min(r1_rho + 1.0 / (float(tau + 1)), 1.0)
            else:
                r1_rho = 0.0

            #  R2 (speed): G[boat_speed <= v_max]
            if sample[0] <= R2_v_max:
                r2_rho = min(r2_rho + 1.0 / (float(tau + 1)), 1.0)
            else:
                r2_rho = 0.0

        # Scale flags in [-0.5, 0.5]
        return r1_rho - 0.5, r2_rho - 0.5