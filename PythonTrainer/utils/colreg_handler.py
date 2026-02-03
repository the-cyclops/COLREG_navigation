
class COLREGHandler:
    """
    Utility to handle the conversion of Unity observations into 
    physical units for COLREG rules evaluation.
    """
    def __init__(self, max_linear_speed=2.5, k_dist=20.0):
        self.max_linear_speed = max_linear_speed
        self.k_dist = k_dist 

    def denormalize(self, obs):
        """
        Converts normalized Unity observations to physical units (m/s and m).
        """
        # Z linearspeed   (Index 6) v/max_speed
        phys_speed = abs(obs[6] * self.max_linear_speed)

        # Distance  (Index 3)
        # Normalized with rational normalization y = d / (k + d)
        y = obs[3]
        phys_dist = (self.k_dist * y) / (1.0 - y + 1e-6)

        return phys_speed, phys_dist