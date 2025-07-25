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


[0. 0. 0.] -7.152557373046875e-07
[0. 0. 0.] -0.01021432876586914
[0. 0. 0.] -0.010130167007446289
[0. 0. 0.] -0.010133743286132812
[0. 0. 0.] -0.010120868682861328
[0. 0. 0.] -0.010149717330932617
[0. 0. 0.] -0.010116815567016602
[0. 0. 0.] -0.010154485702514648
[0. 0. 0.] -0.010123968124389648
[0. 0. 0.] -0.010135412216186523
[0. 0. 0.] -0.010120868682861328
[0. 0. 0.] -0.010144233703613281
[0. 0. 0.] -0.010103940963745117
[0. 0. 0.] -0.010082483291625977
[0. 0. 0.] -0.010078191757202148
[0. 0. 0.] -0.010077953338623047
[0. 0. 0.] -0.010076761245727539
[0. 0. 0.] -0.010073184967041016
[0. 0. 0.] -0.01006937026977539
[0. 0. 0.] -0.01006174087524414
[0. 0. 0.] -0.010135412216186523
[0. 0. 0.] -0.010130643844604492
[0. 0. 0.] -0.010141134262084961
[0. 0. 0.] -0.010157108306884766
[0. 0. 0.] -0.010112524032592773
[0. 0. 0.] -0.010112285614013672
[0. 0. 0.] -0.010109186172485352
[0. 0. 0.] -0.010132312774658203
[0. 0. 0.] -0.010149478912353516
[0. 0. 0.] -0.01013493537902832
[0. 0. 0.] -0.010138750076293945
[0. 0. 0.] -0.010138988494873047
[0. 0. 0.] -0.010150909423828125
[0. 0. 0.] -0.010132312774658203
[0. 0. 0.] -0.010128021240234375
[0. 0. 0.] -0.0101165771484375
[0. 0. 0.] -0.01013493537902832
[0. 0. 0.] -0.010100603103637695
[0. 0. 0.] -0.010124444961547852
[0. 0. 0.] -0.010126590728759766
[0. 0. 0.] -0.010113239288330078
[0. 0. 0.] -0.010115623474121094
[0. 0. 0.] -0.010117530822753906
[0. 0. 0.] -0.01013636589050293
[0. 0. 0.] -0.010123729705810547
[0. 0. 0.] -0.01011037826538086
[0. 0. 0.] -0.010114669799804688
[0. 0. 0.] -0.010103940963745117
[0. 0. 0.] -0.010127544403076172
[0. 0. 0.] -0.010120630264282227
[0. 0. 0.] -0.010114908218383789
[0. 0. 0.] -0.01012420654296875
[0. 0. 0.] -0.010117769241333008
[0. 0. 0.] -0.010116100311279297
[0. 0. 0.] -0.010137796401977539
[0. 0. 0.] -0.010141134262084961
[0. 0. 0.] -0.010126590728759766
[0. 0. 0.] -0.010103702545166016
[0. 0. 0.] -0.01012730598449707
[0. 0. 0.] -0.010101795196533203
[0. 0. 0.] -0.010185003280639648
[0. 0. 0.] -0.010116338729858398
[0. 0. 0.] -0.010128974914550781
[0. 0. 0.] -0.01011347770690918
[0. 0. 0.] -0.010127067565917969
[0. 0. 0.] -0.010125398635864258
[0. 0. 0.] -0.010141372680664062
[0. 0. 0.] -0.010117053985595703
[0. 0. 0.] -0.010118961334228516
[0. 0. 0.] -0.010134220123291016
[0. 0. 0.] -0.01010751724243164
[0. 0. 0.] -0.010123968124389648
[0. 0. 0.] -0.010115861892700195
[0. 0. 0.] -0.010114908218383789
[0. 0. 0.] -0.010121583938598633
[0. 0. 0.] -0.010107040405273438
[0. 0. 0.] -0.010150909423828125
[0. 0. 0.] -0.010115861892700195
[0. 0. 0.] -0.010135412216186523
[0. 0. 0.] -0.01013636589050293
[0. 0. 0.] -0.010127782821655273
[0. 0. 0.] -0.010102987289428711
[0. 0. 0.] -0.01008152961730957
[0. 0. 0.] -0.01009678840637207
[0. 0. 0.] -0.01009058952331543
[0. 0. 0.] -0.010099411010742188
[0. 0. 0.] -0.010076045989990234
[0. 0. 0.] -0.010070085525512695
[0. 0. 0.] -0.010070323944091797
[0. 0. 0.] -0.010119438171386719
[0. 0. 0.] -0.010078907012939453
[0. 0. 0.] -0.010071039199829102
[0. 0. 0.] -0.010083198547363281
[0. 0. 0.] -0.010069847106933594
[0. 0. 0.] -0.010078668594360352
[0. 0. 0.] -0.010071277618408203
[0. 0. 0.] -0.01007080078125
[0. 0. 0.] -0.010081291198730469
[0. 0. 0.] -0.010063409805297852
[0. 0. 0.] -0.010082483291625977
[0. 0. 0.] -0.010068655014038086
[0. 0. 0.] -0.010065793991088867
[0. 0. 0.] -0.01006627082824707
[0. 0. 0.] -0.010084152221679688
[0. 0. 0.] -0.010091304779052734
[0. 0. 0.] -0.010065793991088867
[0. 0. 0.] -0.010065793991088867
[0. 0. 0.] -0.010064840316772461
[0. 0. 0.] -0.010071516036987305
[0. 0. 0.] -0.010086536407470703
[0. 0. 0.] -0.01007533073425293
[0. 0. 0.] -0.010064125061035156
[0. 0. 0.] -0.01006627082824707
[0. 0. 0.] -0.01007533073425293
[0. 0. 0.] -0.010063648223876953
[0. 0. 0.] -0.010083198547363281
[0. 0. 0.] -0.010074853897094727
[0. 0. 0.] -0.010067224502563477
[0. 0. 0.] -0.01006627082824707
[0. 0. 0.] -0.010102987289428711
[0. 0. 0.] -0.010065317153930664
[0. 0. 0.] -0.010078907012939453
[0. 0. 0.] -0.010075569152832031
[0. 0. 0.] -0.010065555572509766
[0. 0. 0.] -0.010084867477416992
[0. 0. 0.] -0.010127544403076172
[0. 0. 0.] -0.010123968124389648
[0. 0. 0.] -0.01012873649597168
[0. 0. 0.] -0.01011037826538086
[0. 0. 0.] -0.010137081146240234
[0. 0. 0.] -0.010121822357177734
[0. 0. 0.] -0.010141611099243164
[0. 0. 0.] -0.01013326644897461
[0. 0. 0.] -0.010069608688354492
[0. 0. 0.] -0.010081052780151367
[0. 0. 0.] -0.010085344314575195
[0. 0. 0.] -0.01006937026977539
[0. 0. 0.] -0.010079383850097656
[0. 0. 0.] -0.010074377059936523
[0. 0. 0.] -0.010063886642456055
[0. 0. 0.] -0.010070323944091797
[0. 0. 0.] -0.010084390640258789
[0. 0. 0.] -0.010081291198730469
[0. 0. 0.] -0.010065555572509766
[0. 0. 0.] -0.01006317138671875
[0. 0. 0.] -0.010071516036987305
[0. 0. 0.] -0.01012873649597168
[0. 0. 0.] -0.0100860595703125
[0. 0. 0.] -0.010069608688354492
[0. 0. 0.] -0.01009988784790039
[0. 0. 0.] -0.010069608688354492
[0. 0. 0.] -0.010121822357177734
[0. 0. 0.] -0.010162115097045898
[0. 0. 0.] -0.010127067565917969
[0. 0. 0.] -0.010084867477416992
[0. 0. 0.] -0.01009368896484375
[0. 0. 0.] -0.010068416595458984
[0. 0. 0.] -0.010101318359375
[0. 0. 0.] -0.010106563568115234
[0. 0. 0.] -0.010066986083984375
[0. 0. 0.] -0.010074853897094727
[0. 0. 0.] -0.01009821891784668
[0. 0. 0.] -0.010065078735351562
[0. 0. 0.] -0.010085105895996094
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010092973709106445
[0. 0. 0.] -0.010069131851196289
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010095834732055664
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010140657424926758
[0. 0. 0.] -0.01011967658996582
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010088443756103516
[0. 0. 0.] -0.010156631469726562
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010121583938598633
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010133743286132812
[0. 0. 0.] -0.010118484497070312
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010104894638061523
[0. 0. 0.] -0.010129690170288086
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010113716125488281
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010122537612915039
[0. 0. 0.] -0.010117530822753906
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.0101318359375
[0. 0. 0.] -0.010128498077392578
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010121345520019531
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010132551193237305
[0. 0. 0.] -0.01018381118774414
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010135412216186523
[0. 0. 0.] -0.010113716125488281
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010115623474121094
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010111093521118164
[0. 0. 0.] -0.010089397430419922
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010129213333129883
[0. 0. 0.] -0.010087013244628906
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010085582733154297
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010088205337524414
[0. 0. 0.] -0.010098695755004883
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010085821151733398
[0. 0. 0.] -0.010083436965942383
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010083198547363281
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010097742080688477
[0. 0. 0.] -0.010093212127685547
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010089874267578125
[0. 0. 0.] -0.010082244873046875
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010101795196533203
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.01004338264465332
[0. 0. 0.] -0.010082721710205078
Data processing error: 'SensorComm' object has no attribute 'predict_force'
[0. 0. 0.] -0.010089874267578125
[0. 0. 0.] -0.010085105895996094
