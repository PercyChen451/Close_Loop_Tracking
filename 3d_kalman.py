import numpy as np
from scipy.stats import multivariate_normal

class GaussianKalmanFilter3D:
    def __init__(self, initial_pos, initial_uncertainty=100.0, process_noise=None, measurement_noise=None):
        """
        3D Gaussian Kalman Filter with constant velocity model
        
        Args:
            initial_pos: (x, y, z) starting position (tuple/list/np.array)
            initial_uncertainty: Initial covariance scaling factor
            process_noise: Optional 6x6 process noise matrix (Q)
            measurement_noise: Optional 3x3 measurement noise matrix (R)
        """
        initial_pos = np.array(initial_pos, dtype=np.float32)
        
        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.concatenate([initial_pos, [0, 0, 0]]).astype(np.float32)
        
        # 6x6 state covariance matrix
        self.covariance = np.eye(6, dtype=np.float32) * initial_uncertainty
        
        # Process noise matrix (Q)
        self.Q = process_noise if process_noise is not None else np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0]).astype(np.float32)
        
        # Measurement noise matrix (R)
        self.R = measurement_noise if measurement_noise is not None else np.eye(3, dtype=np.float32) * 5.0
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(6, dtype=np.float32)
        self.F[:3, 3:6] = np.eye(3)  # Position += velocity
        
        # Measurement matrix (observing position only)
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[:3, :3] = np.eye(3)
        
        # Belief distribution (now 3D)
        self.belief = None
        self.update_belief()
        
        # Velocity boosting
        self.velocity_boost_factor = 0.3
        self.last_measurement = None
        self.last_prediction = None

    def update_belief(self):
        """Update 3D Gaussian belief about current state"""
        self.belief = multivariate_normal(
            mean=self.state[:3],
            cov=self.covariance[:3, :3]
        )

    def predict(self):
        """Predict next state using state transition"""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        self.last_prediction = self.state.copy()
        self.update_belief()
        return self.state[:3]

    def update(self, measurement):
        """
        Update state with new 3D measurement
        
        Args:
            measurement: (x, y, z) position or None
            
        Returns:
            Filtered (x, y, z) position
        """
        if measurement is None:
            return self.predict()
            
        measurement = np.array(measurement, dtype=np.float32)
        self.last_measurement = measurement.copy()
        
        # Prediction step
        predicted_state = self.F @ self.state
        predicted_cov = self.F @ self.covariance @ self.F.T + self.Q
        
        # Measurement update
        y = measurement - self.H @ predicted_state
        S = self.H @ predicted_cov @ self.H.T + self.R
        K = predicted_cov @ self.H.T @ np.linalg.inv(S)
        
        self.state = predicted_state + K @ y
        self.covariance = (np.eye(6) - K @ self.H) @ predicted_cov
        self.update_belief()
        return self.state[:3]

    def probability(self, position):
        """
        Get probability of given 3D position under current belief
        
        Args:
            position: (x, y, z) position to evaluate
            
        Returns:
            Probability density at given position
        """
        if self.belief is None:
            return 0.0
        return self.belief.pdf(position)

    def get_boosted_position(self):
        """Get 3D position with velocity boost applied"""
        return self.state[:3] + self.velocity_boost_factor * self.state[3:6]

    def adaptive_update(self, measurement, movement_threshold=30, prob_threshold=0.01):
        """
        Smart update that adapts to 3D movement speed and measurement probability
        
        Args:
            measurement: (x, y, z) position or None
            movement_threshold: Speed (units/frame) for fast movement
            prob_threshold: Minimum probability for full trust
            
        Returns:
            Filtered 3D position, possibly with velocity boost
        """
        if measurement is None:
            return self.predict()
            
        measurement = np.array(measurement, dtype=np.float32)
        
        # Calculate 3D movement speed
        movement_speed = 0.0
        if self.last_measurement is not None:
            movement_speed = np.linalg.norm(measurement - self.last_measurement)
        
        # Get probability of current measurement
        prob = self.probability(measurement)
        
        # Adaptive logic
        if movement_speed > movement_threshold or prob > prob_threshold:
            self.update(measurement)
            return self.get_boosted_position()
        else:
            return self.update(measurement)
