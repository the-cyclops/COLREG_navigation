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

    
    def compute_markovian_flags(self):

        if not self.robustness_1 or not self.robustness_2:
            return 0.5, 0.5

        tau = self.stl_window.maxlen
        
        recent_r1 = self.robustness_1[-tau:] 
        recent_r2 = self.robustness_2[-tau:]

        missing_samples = tau - len(recent_r1)      
        step_increment = 1.0 / float(tau + 1)

        r1_flag = min(missing_samples * step_increment, 1)
        r2_flag = min(missing_samples * step_increment, 1)

        for rho in recent_r1:
            if rho >=0:
                r1_flag = min(r1_flag + step_increment, 1)
            else:
                r1_flag = 0.0

        for rho in recent_r2:
            if rho >=0:
                r2_flag = min(r2_flag + step_increment, 1)
            else:
                r2_flag = 0.0

        # Scale flags in [-0.5, 0.5]
        return r1_flag - 0.5, r2_flag - 0.5