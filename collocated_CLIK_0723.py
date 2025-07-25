def main():
    np.set_printoptions(precision=3, suppress=True)
    
    # Initialize sensor first with longer timeout
    print("Initializing force sensor...")
    sen_connect = SensorConnect('/dev/ttyUSB0', 115200)
    sensor = SensorComm(sen_connect)
    
    # Wait for sensor to stabilize
    time.sleep(2)
    
    # Perform calibration
    print("Calibrating force sensor...")
    sensor.calibrate_sensor()
    
    # Initialize other components
    print("Initializing Arduino...")
    connect = ArduinoConnect('/dev/ttyACM0', 250000)
    arduino = ArduinoComm(connect)
    time.sleep(1)
    
    print("Initializing camera tracker...")
    tracker = Cam_Tracker()
    dyn = CollocatedDynamics()
    
    # Setup trajectory
    path_coords = dyn.generate_circle(15, 30, 5)
    
    # Main control loop
    try:
        while True:
            # Get sensor data
            force = sensor.receive_data()
            
            # Get robot data
            robot_data = arduino.receive_data()
            
            # Update tracker
            tracker.update(rigid_pred)
            t_tip, R_tip = tracker.get_pose()
            
            # Run control algorithm
            (u_volumes, commands, backbone_pred, 
             rigid_pred, tip_pred, dt) = dyn.CLIK(
                robot_data, t_tip*1000, R_tip, force, path_coords
            )
            
            # Send commands
            arduino.send_data(u_volumes, commands)
            
            time.sleep(0.005)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        arduino.send_data(np.zeros(3), [0, 0, 0, 0])
        sensor.close()
        connect.close()
        print("Resources released")


class SensorComm:
    def __init__(self, chain):
        self.comm = chain
        self.initialized = False
        self.current_data = np.array([0, 0, 0], dtype=np.float32)
        
        # Load model and normalization parameters
        self.load_model_and_params()
        
        # Initialize buffers
        self.init_buffers()
        
        # Start separate thread for reading sensor data
        self.start_reader_thread()

    def load_model_and_params(self):
        """Load model and normalization parameters with type checking"""
        self.norm_params = np.load('normalization_params.npy', allow_pickle=True).item()
        self.X_mean = self.norm_params['mean'].astype(np.float32)
        self.X_std = self.norm_params['std'].astype(np.float32)
        self.Y_mean = self.norm_params.get('Y_mean', 0).astype(np.float32)
        self.Y_std = self.norm_params.get('Y_std', 1).astype(np.float32)
        
        self.model = torch.jit.load('force_calibration_model_optimized.pt')
        self.model.eval()
        self.model.float()  # Ensure model uses float32

    def init_buffers(self):
        """Initialize data buffers"""
        self.SMOOTHING_WINDOW = 5
        self.bx_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.by_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bz_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bx2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.by2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bz2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)

    def start_reader_thread(self):
        """Start background thread for continuous reading"""
        import threading
        self.reader_active = True
        self.reader_thread = threading.Thread(target=self.continuous_reader)
        self.reader_thread.daemon = True
        self.reader_thread.start()

    def continuous_reader(self):
        """Background thread for reading sensor data"""
        while self.reader_active:
            try:
                if self.comm.in_waiting > 0:
                    line = self.comm.readline().decode('utf-8').strip()
                    self.process_line(line)
            except Exception as e:
                print(f"Reader thread error: {e}")
                time.sleep(0.1)

    def process_line(self, line):
        """Process a single line of sensor data"""
        # Skip initialization messages and headers
        if not line or any(x in line for x in ['Initializing', 'sensor', 'X1', 'Y1', 'Z1']):
            return

        try:
            # Clean and parse data
            cleaned = ''.join([c for c in line if c in '0123456789,.- '])
            values = [float(x.strip()) for x in cleaned.split(',') if x.strip()]
            
            if len(values) == 6:
                bx, by, bz, bx2, by2, bz2 = values
                self.update_buffers(bx, by, bz, bx2, by2, bz2)
                force_vec = self.predict_force()
                self.current_data = force_vec - self.baseForce if hasattr(self, 'baseForce') else force_vec
        except Exception as e:
            print(f"Data processing error: {e}")

    def update_buffers(self, bx, by, bz, bx2, by2, bz2):
        """Update smoothing buffers"""
        self.bx_buffer.append(bx)
        self.by_buffer.append(by)
        self.bz_buffer.append(bz)
        self.bx2_buffer.append(bx2)
        self.by2_buffer.append(by2)
        self.bz2_buffer.append(bz2)

    def calibrate_sensor(self, duration=5.0):
        """Perform sensor calibration"""
        print("Starting calibration - keep sensor at rest...")
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if len(samples) >= 500:  # Maximum samples to collect
                break
            time.sleep(0.01)
        
        if samples:
            X_cal = np.array(samples)
            X_norm = (X_cal - self.X_mean) / self.X_std
            with torch.no_grad():
                Y_norm = self.model(torch.from_numpy(X_norm).float()).numpy()
            self.baseForce = np.mean(Y_norm * self.Y_std + self.Y_mean, axis=0)
            print(f"Calibration complete. Base force: {self.baseForce}")
        else:
            print("Warning: No valid calibration data collected")
            self.baseForce = np.array([0, 0, 0], dtype=np.float32)
        
        return self.baseForce

    def receive_data(self):
        """Get the latest force data"""
        return self.current_data.copy()

    def close(self):
        """Cleanup resources"""
        self.reader_active = False
        self.reader_thread.join()
        self.comm.close()



