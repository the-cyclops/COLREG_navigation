import numpy as np

class COLREGHandler:
    def __init__(self, max_linear_speed=2.5):
        self.max_linear_speed = max_linear_speed
        # Parameters of Unity C# normalization
        self.k_dist = 7.0 
        self.k_intruder_vel_rel = 2.0 * max_linear_speed 
        
        # Constant for safety signal clipping (Avoids magic numbers)
        self.MAX_SAFETY_MARGIN_CAP = 1.0

    def get_ego_speed(self, obs_vector):
        """
        Extracts and denormalizes the Ego boat linear speed (Rule R2).
        """
        # Indices 6, 7, 8 are the normalized local velocity vector
        norm_vel_vector = obs_vector[6:9]
        norm_speed = np.linalg.norm(norm_vel_vector)
        phys_speed = norm_speed * self.max_linear_speed
        return phys_speed

    def denormalize_intruder_observation(self, obs_vector):
        """
        Extracts and denormalizes intruder data from the observation vector.
        """
        # --- Intruder 1 ---
        dir1 = obs_vector[10:13] 
        raw_dist1 = obs_vector[13]
        
        if raw_dist1 > 0.99:
            dist1 = 100.0
        else:
            # Denormalization of rational function: dist = (k * raw) / (1 - raw)
            dist1 = (self.k_dist * raw_dist1) / (1.0 - raw_dist1 + 1e-6)
            
        pos_rel1 = dir1 * dist1
        vel_rel1 = obs_vector[14:17] * self.k_intruder_vel_rel

        # --- Intruder 2 ---
        dir2 = obs_vector[17:20]
        raw_dist2 = obs_vector[20]
        if raw_dist2 > 0.99:
            dist2 = 100.0
        else:
            dist2 = (self.k_dist * raw_dist2) / (1.0 - raw_dist2 + 1e-6)
            
        pos_rel2 = dir2 * dist2
        vel_rel2 = obs_vector[21:24] * self.k_intruder_vel_rel

        return [(pos_rel1, vel_rel1), (pos_rel2, vel_rel2)]

    def compute_cpa_R1(self, pos_rel, vel_rel, safe_dist=1.0, t_horizon=10.0):
        """
        Calculates safety signal based on CPA (Closest Point of Approach) over t_horizon seconds.
        Returns: R1 Signal (Predicted Min Distance - Safety Distance)
        Interpretation: possitive values indicate safety margin, negative values indicate violation.
        """
        # --- PHYSICS CALCULATION (Real world units) ---

        # If visually padding (distance > 500m), skip physics logic
        if np.linalg.norm(pos_rel) > 500.0:
            return self.MAX_SAFETY_MARGIN_CAP

        dv2 = np.dot(vel_rel, vel_rel)
        
        if dv2 < 1e-6:
            t_cpa = 0.0
        else:
            t_cpa = -np.dot(pos_rel, vel_rel) / dv2
        
        # 1. Diverging (Moving away) -> Safety signal based on current distance
        if t_cpa < 0:
            min_dist = np.linalg.norm(pos_rel)
        # 2. Converging slowly (Risk beyond horizon) -> Safety signal based on distance at horizon
        elif t_cpa > t_horizon:
            pos_at_horizon = pos_rel + vel_rel * t_horizon
            min_dist = np.linalg.norm(pos_at_horizon)
        # 3. Converging fast (Risk imminent) -> Safety signal based on CPA distance
        else:
            pos_cpa = pos_rel + vel_rel * t_cpa
            min_dist = np.linalg.norm(pos_cpa)

        # --- OUTPUT CLIPPING ---
        
        # This is the "uncapped" safety signal (can be negative for violations, positive for safe)
        raw_margin = min_dist - safe_dist
        
        # Apply strict clipping.
        # Negative values (violations) are preserved as-is.
        # Positive values (safe) are capped to stabilize Value Network training.
        return min(raw_margin, self.MAX_SAFETY_MARGIN_CAP)

    def get_R1_safety_signal(self, obs, safe_dist=1.0):
        """
        Main function to call in the training loop for Rule R1.
        Returns the worst (minimum) signal value, which will be used by rtamt for robustness calculation.
        """
        intruders_data = self.denormalize_intruder_observation(obs)
        signals = []
        
        for pos, vel in intruders_data:
            # Calculate individual safety signal per intruder
            signal = self.compute_cpa_R1(pos, vel, safe_dist=safe_dist)
            signals.append(signal)
            
        # Return the critical safety signal (the lowest one)
        return min(signals)