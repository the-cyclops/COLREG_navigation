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

        # Sliding window for temporal logic evaluation (physical data)
        self.stl_window = deque(maxlen=stl_horizon)

    def add_ppo_transition(self, state, action, logprob, reward, is_terminal):
        """Store transition data for PPO update."""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def add_stl_sample(self, phys_speed, r1_robustness):
        """Add denormalized physical data to the sliding window."""
        self.stl_window.append([phys_speed, r1_robustness])

    def add_costs(self, c_r1, c_r2):
        """Store costs derived from RTAMT robustness values."""
        self.cost_r1.append(c_r1)
        self.cost_r2.append(c_r2)

    def clear_ppo(self):
        """Clear on-policy buffers after policy update."""
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]
        del self.cost_r1[:]
        del self.cost_r2[:]

    def clear_stl_window(self):
        """Clear temporal window at the beginning of each episode."""
        self.stl_window.clear()

    def is_stl_ready(self):
        """Check if the buffer has enough samples for STL evaluation."""
        return len(self.stl_window) == self.stl_window.maxlen