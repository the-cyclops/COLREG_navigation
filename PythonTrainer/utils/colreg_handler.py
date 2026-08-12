import numpy as np

class COLREGHandler:
    def __init__(self, max_linear_speed=2.5):
        self.max_linear_speed = max_linear_speed
        # Parameters of Unity C# normalization
        self.max_dist = 43.0 
        self.k_intruder_vel_rel = 2.0 * max_linear_speed 
        
        # Constant for safety signal clipping (Avoids magic numbers)
        self.MAX_SAFETY_MARGIN_CAP = 1.0

    def get_ego_speed(self, obs_vector):
        """
        Extracts and denormalizes the Ego boat linear speed (Rule R2).
        """
        # Indices 3 is normalized velocity x, 4 is normalized velocity z
        norm_speed = obs_vector[4]
        phys_speed = norm_speed * self.max_linear_speed
        return phys_speed

    def denormalize_intruder_observation(self, obs_vector):
        """
        Extracts and denormalizes intruder data from the observation vector.
        """
        # --- Intruder 1 ---
        dir1 = obs_vector[6:8] 
        raw_dist1 = obs_vector[8]
        lin_vel1 = obs_vector[9:11]
        
        if raw_dist1 > 0.99:
            # intruder really far or absent
            # really high dist to trigger guard in CPA calculation
            pos_rel1 = np.array([999.0, 999.0])
            vel_rel1 = np.array([0.0, 0.0])
        else:
            # Denormalization 
            dist1 = (raw_dist1 * self.max_dist) 
            pos_rel1 = dir1 * dist1
            vel_rel1 = lin_vel1 * self.k_intruder_vel_rel

        # --- Intruder 2 ---
        dir2 = obs_vector[13:15]
        raw_dist2 = obs_vector[15]
        lin_vel2 = obs_vector[16:18]

        if raw_dist2 > 0.99:
            # intruder really far or absent
            pos_rel2 = np.array([999.0, 999.0])
            vel_rel2 = np.array([0.0, 0.0])
        else:
            dist2 = (raw_dist2 * self.max_dist) 
            pos_rel2 = dir2 * dist2
            vel_rel2 = lin_vel2 * self.k_intruder_vel_rel

        return [(pos_rel1, vel_rel1), (pos_rel2, vel_rel2)]

    def compute_cpa_R1(self, pos_rel, vel_rel, safe_dist=2.0, t_horizon=5.0):
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

    def get_R1_safety_signal(self, obs, safe_dist=2.0, t_coll=5.0):
        """
        Main function to call in the training loop for Rule R1.
        Returns the worst (minimum) signal value, which will be used by rtamt for robustness calculation.
        """
        intruders_data = self.denormalize_intruder_observation(obs)
        signals = []
        
        for pos, vel in intruders_data:
            # Calculate individual safety signal per intruder
            signal = self.compute_cpa_R1(pos, vel, safe_dist=safe_dist, t_horizon=t_coll)
            signals.append(signal)
            
        # Return the critical safety signal (the lowest one)
        return min(signals)

    def get_keep_signal(self, obs, safe_dist=2.0, t_check=10.0, delta_head_on=5.0, max_left_angle=112.5): # 112.5 is the 180-degreeequivalent of 247.5 degrees in 360-degree system
        """
        Calculates the 'keep' signal (Stand-on vessel status) using STL robustness semantics.
        Logic: Collision Risk AND Intruder in Left Sector (between delta_head_on and max_left_angle)
        Returns > 0 if the rule is active (i.e., you must keep your course).
        """
        intruders_data = self.denormalize_intruder_observation(obs)

        pos, vel = intruders_data[0]  # Only consider the first intruder for the keep signal

        # Skip padded/dummy intruders
        if np.linalg.norm(pos) > 500.0:
            return -1.0  # No intruder, rule not active

        # 1. COLLISION RISK SIGNAL (In meters)
        # Collision risk is positive if the intruder is predicted to violate the safe distance within the time horizon (cpa_margin < 0).
        cpa_margin = self.compute_cpa_R1(pos, vel, safe_dist, t_check)
        collision_risk_signal = -cpa_margin 

        # 2. EXACT LEFT SECTOR SIGNAL (Angular)
        # pos[0] is X (Right), pos[1] is Z (Forward).
        # By using -pos[0], np.arctan2 maps the left side to positive degrees [0, 180]
        # and the right side to negative degrees [0, -180].
        angle_left_deg = np.degrees(np.arctan2(-pos[0], pos[1]))

        # STL Robustness for being within [delta_head_on, max_left_angle]
        # This is positive ONLY if angle_left_deg is strictly inside the sector boundaries.
        raw_angular_robustness = min(angle_left_deg - delta_head_on, max_left_angle - angle_left_deg)
            
        # Scale the angular signal so it matches the magnitude of the CPA distance signal.
        # An angle margin of 50 degrees becomes a robustness of 5.0 (similar to 5.0 meters).
        scaling_factor = 10.0  # degrees to meters scaling #TODO controllare che sia fintunato
        left_sector_signal = raw_angular_robustness / scaling_factor

        # 3. LOGICAL AND (min)
        # The rule is active ONLY IF there is a risk AND it's strictly in the left sector.
        intruder_keep = min(collision_risk_signal, left_sector_signal)
        return intruder_keep


    def get_no_turning_signal(self, steering_action, steering_threshold=0.1):
        """
        Returns > 0 if the ego vessel is maintaining its course.
        Evaluates the agent's steering action directly instead of physical inertia.
        """
        # steering_action is in the [-1.0, 1.0] range.
        # The signal is positive if the absolute steering command is below the threshold.
        no_turning_robustness = steering_threshold - abs(steering_action)
        
        return float(no_turning_robustness)