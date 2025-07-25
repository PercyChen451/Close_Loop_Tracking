def receive_data(self):
    data_buffer = []
    
    while self.comm.in_waiting > 0:
        try:
            line = self.comm.readline().decode('utf-8').strip()
            
            # Skip initialization messages and headers
            if not line or 'Initializing' in line or 'sensors' in line or 'X1' in line:
                continue
                
            # Clean the line by removing any non-numeric characters except commas and minus signs
            cleaned_line = ''.join([c for c in line if c in '0123456789,.- '])
            line_v = cleaned_line.split(",")
            
            # Ensure we have exactly 6 values
            if len(line_v) != 6:
                continue
                
            bx, by, bz, bx2, by2, bz2 = [float(val.strip()) for val in line_v]

            # Apply smoothing
            self.bx_buffer.append(bx)
            self.by_buffer.append(by)
            self.bz_buffer.append(bz)
            self.bx2_buffer.append(bx2)
            self.by2_buffer.append(by2)
            self.bz2_buffer.append(bz2)

            smoothed_bx = np.mean(self.bx_buffer) if self.bx_buffer else bx
            smoothed_by = np.mean(self.by_buffer) if self.by_buffer else by
            smoothed_bz = np.mean(self.bz_buffer) if self.bz_buffer else bz
            smoothed_bx2 = np.mean(self.bx2_buffer) if self.bx2_buffer else bx2
            smoothed_by2 = np.mean(self.by2_buffer) if self.by2_buffer else by2
            smoothed_bz2 = np.mean(self.bz2_buffer) if self.bz2_buffer else bz2

            # Predict force
            force_vec = self.predict_force(
                smoothed_bx, smoothed_by, smoothed_bz,
                smoothed_bx2, smoothed_by2, smoothed_bz2
            )
            
            if self.baseForce is not None:
                data_buffer.append(force_vec - self.baseForce)
            else:
                data_buffer.append(force_vec)
                
        except (ValueError, IndexError) as e:
            print(f"Skipping malformed data: {line}")
            continue
    
    if len(data_buffer) > 0:
        self.current_data = data_buffer[-1]
    
    return self.current_data if self.current_data is not None else np.array([0, 0, 0])






def main():
    sen_con = SensorConnect('/dev/ttyUSB0', 115200)
    sen = SensorComm(sen_con)
    
    # Wait for initialization to complete
    time.sleep(2)  
    
    # Clear any buffered initialization messages
    while sen_con.in_waiting > 0:
        sen_con.readline()
    
    # Now perform calibration
    sen.baseForce = sen.calibrate_sensor()
    
    # Main loop
    while True:
        force = sen.receive_data()
        print(force)
        time.sleep(0.01)
