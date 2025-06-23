class GaussianKalmanFilter3D:
    def __init__(self, initial_pos, initial_uncertainty=100.0, 
                 process_noise=None, measurement_noise=None):
        """Initialize 3D Kalman Filter"""
        initial_pos = np.array(initial_pos, dtype=np.float32)
        
        # State: [x, y, z, vx, vy, vz]
        self.state = np.concatenate([initial_pos, [0, 0, 0]]).astype(np.float32)
        
        # 6x6 covariance matrix
        self.covariance = np.eye(6, dtype=np.float32) * initial_uncertainty
        
        # Process noise (6x6)
        self.Q = process_noise if process_noise is not None else np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
        
        # Measurement noise (3x3)
        self.R = measurement_noise if measurement_noise is not None else np.eye(3, dtype=np.float32) * 5.0
        
        # State transition matrix (6x6)
        self.F = np.eye(6, dtype=np.float32)
        self.F[:3, 3:6] = np.eye(3)  # Position += velocity
        
        # Measurement matrix (3x6)
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[:3, :3] = np.eye(3)  # Observe position only
        
        self.velocity_boost_factor = 0.3
        self.last_measurement = None
        self.last_prediction = None
    
    def predict(self):
        """Predict next state in 3D"""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        self.last_prediction = self.state.copy()
        return self.state[:3]
    
    def update(self, measurement):
        """Update with new 3D measurement"""
        if measurement is None:
            return self.predict()
            
        measurement = np.array(measurement, dtype=np.float32)
        self.last_measurement = measurement.copy()
        
        # Prediction step
        predicted_state = self.F @ self.state
        predicted_cov = self.F @ self.covariance @ self.F.T + self.Q
        
        # Update step
        y = measurement - self.H @ predicted_state
        S = self.H @ predicted_cov @ self.H.T + self.R
        K = predicted_cov @ self.H.T @ np.linalg.inv(S)
        
        self.state = predicted_state + K @ y
        self.covariance = (np.eye(6) - K @ self.H) @ predicted_cov
        return self.state[:3]
    
    def get_boosted_position(self):
        """Get position with 3D velocity boost"""
        return self.state[:3] + self.velocity_boost_factor * self.state[3:6]