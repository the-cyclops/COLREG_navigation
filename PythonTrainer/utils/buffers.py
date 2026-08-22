class Memory:
    def __init__(self, tau=80):

        self.tau = tau

        # PPO buffers for on-policy training
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []
        
        # COLREG cost buffers populated via RTAMT robustness
        self.cost_r1 = []
        self.cost_r2 = []
        self.cost_r6 = [] 

        # Additional buffers to decide what to optiminze for in the PPO loss function
        self.robustness_1 = []
        self.robustness_2 = []
        self.robustness_6 = []

        # This window is now obslete, was used for call to rtamt during every step and not just at the end of the episode
        # Sliding window for temporal logic evaluation (physical data)
        #self.stl_window = deque(maxlen=stl_horizon)
        #self.clear_stl_window()

        # Buffers to store episode data to be used for post-episode RTAMT evaluation
        self.episode_r1_signal = []
        self.episode_phys_speed = []
        self.episode_keep_signal = []
        self.episode_no_turning_signal = []

    def add_ppo_transition(self, state, action, logprob, reward, is_terminal, phys_speed, r1_signal, keep_signal, no_turn_signal):
        """Store transition data for PPO update and denormalized physical data of the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.episode_phys_speed.append(phys_speed)
        self.episode_r1_signal.append(r1_signal)
        self.episode_keep_signal.append(keep_signal)
        self.episode_no_turning_signal.append(no_turn_signal)

    def add_costs(self, c_r1, c_r2, c_r6):
        """Store costs derived from RTAMT robustness values."""
        self.cost_r1.extend(c_r1)
        self.cost_r2.extend(c_r2)
        self.cost_r6.extend(c_r6)

    def add_robustness(self, r1, r2, r6):
        """Store robustness values for later analysis."""
        self.robustness_1.extend(r1)
        self.robustness_2.extend(r2)
        self.robustness_6.extend(r6)

    def clear_ppo(self):
        """Clear on-policy buffers after policy update."""
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]
        del self.cost_r1[:]
        del self.cost_r2[:]
        del self.cost_r6[:]
        del self.robustness_1[:]
        del self.robustness_2[:]
        del self.robustness_6[:]
    
    def clear_episode_signal(self):
        """Clear temporal lists at the end of each episode."""
        del self.episode_phys_speed[:]
        del self.episode_r1_signal[:]
        del self.episode_keep_signal[:]
        del self.episode_no_turning_signal[:]
    
    def compute_markovian_flags(self, v_max=2.1):

        if not self.episode_r1_signal or not self.episode_phys_speed or not self.episode_keep_signal or not self.episode_no_turning_signal:
            return 0.5, 0.5, 0.5
        
        tau = self.tau
        
        # take last tau samples
        # > 0.0 if there is no immediate collision risk (within t_coll)
        recent_r1s = self.episode_r1_signal[-tau:] if self.episode_r1_signal else []
        # Physical linear speed of the ego vessel (m/s)
        recent_speeds = self.episode_phys_speed[-tau:] if self.episode_phys_speed else []
        # > 0.0 if ego is the Stand-on vessel (intruder in left sector and risk within t_check)
        recent_keeps = self.episode_keep_signal[-tau:] if self.episode_keep_signal else []
        # >= 0.0 if the ego vessel is strictly maintaining its heading
        recent_no_turns = self.episode_no_turning_signal[-tau:] if self.episode_no_turning_signal else []

        missing_samples = tau - len(recent_r1s)      
        step_increment = 1.0 / float(tau + 1)

        r1_flag = min(missing_samples * step_increment, 1.0)
        r2_flag = min(missing_samples * step_increment, 1.0)
        r6_flag = min(missing_samples * step_increment, 1.0)

        # Iterate over the recent samples and update flags based on conditions
        for r1, speed, keep, no_turn in zip(recent_r1s, recent_speeds, recent_keeps, recent_no_turns):
            
            # R1 is safe if the signal is non-negative (globally)
            if r1 >= 0.0:
                r1_flag = min(r1_flag + step_increment, 1.0)
            else:
                r1_flag = 0.0

            # R2 is safe if the speed is within the limits (globally)
            if -1.0 <= speed <= v_max:
                r2_flag = min(r2_flag + step_increment, 1.0)
            else:
                r2_flag = 0.0

            # R6: Satisfied if the ego vessel is either not required to stand-on, or is correctly maintaining its course
            if keep <= 0.0 or no_turn >= 0.0:
                r6_flag = min(r6_flag + step_increment, 1.0)
            else:
                r6_flag = 0.0

        # Scale flags in [-0.5, 0.5]
        return r1_flag - 0.5, r2_flag - 0.5, r6_flag - 0.5

    # old function that was used to compute flags based on robustness values
    # not usable anymore since we now compute robustness at the end of the episode and not at every step
    def ___OLD_compute_markovian_flags(self):

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

    #def clear_stl_window(self):
    #    """Clear temporal window at the beginning of each episode."""
    #    self.stl_window.clear()
    #    # Initialize with safe defaults, 1.0 is the same as MAX_SAFETY_MARGIN_CAP in colreg_handler.py
    #    for _ in range(self.stl_window.maxlen):
    #        self.stl_window.append([0.0, 1.0]) 