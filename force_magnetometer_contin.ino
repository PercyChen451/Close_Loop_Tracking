#include <Wire.h>
#include "fcnn_weights.h"
// Sensor configuration
const byte I2CAddress = 0x60; // Default address for MLX90394
const long freq = 1000000;     // I2C Frequency in Hz
const float mu[3] = {208.812205008607,	-45.5994225109667,	14.5227941584763};
const float sigma[3] = {202.115205105661,	93.0846055668491,	67.7296408356276};
const int base_x = 1715, base_y = 2194, base_z = 106;
// const int base_x = 0, base_y = -0, base_z = -0;
const float delay_millis = 10; // in milliseconds

// Register addresses
#define REG_STAT1 0x00
#define REG_X_L 0x01
#define REG_X_H 0x02
#define REG_Y_L 0x03
#define REG_Y_H 0x04
#define REG_Z_L 0x05
#define REG_Z_H 0x06
#define REG_STAT2 0x07
#define REG_CTRL1 0x0E
char Cmd = 'A';       // Char received from the PC.
int Nack = 0;         // >0 If the IC did not send an Ack.
int Nbytes = 0;       // Number of bytes to read from IC.
int Flag = 0;         // Flag to indicate termination of burst writing to the IC.
// Data variables
int16_t x_val = 0, y_val = 0, z_val = 0;
uint8_t stat1 = 0, stat2 = 0;
int byte_val;
byte vals[] = {0,0,0,0,0,0};
// moving avg filter
const int numReadings  = 15;
int read_x [numReadings], read_y [numReadings], read_z [numReadings];
float total_x  = 0, total_y = 0, total_z = 0;
float avg_x, avg_y, avg_z;
int readIndex  = 0;


bool checkSensorConnection() {
  Wire.beginTransmission(I2CAddress);
  return (Wire.endTransmission() == 0);
}

bool readRegister(uint8_t reg, uint8_t* value) {
  Wire.beginTransmission(I2CAddress);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  
  Wire.requestFrom(I2CAddress, (uint8_t)1);
  if (Wire.available()) {
    *value = Wire.read();
    return true;
  }
  return false;
}

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(I2CAddress);
  Wire.write(reg);
  Wire.write(value);
  return (Wire.endTransmission() == 0);
}

bool dataReady() {
  return readRegister(REG_STAT1, &stat1) && (stat1 & 0x01);
}

bool readSensorData() {
  // 1. Check if data is ready
  if (!dataReady()) {
    return false;
  }

  // 2. Read measurement data (6 bytes: X_L, X_H, Y_L, Y_H, Z_L, Z_H)
  Wire.beginTransmission(I2CAddress);
  Wire.write(REG_X_L);
  if (Wire.endTransmission(false) != 0) return false;
  
  uint8_t data[6];
  Wire.requestFrom(I2CAddress, (uint8_t)6);
  for (int i = 0; i < 6; i++) {
    if (!Wire.available()) return false;
    data[i] = Wire.read();
  }

  // 3. Read STAT2 to complete the sequence
  readRegister(REG_STAT2, &stat2);

  // Convert data to 16-bit values
  x_val = (int16_t)((data[1] << 8) | data[0]);
  y_val = (int16_t)((data[3] << 8) | data[2]);
  z_val = (int16_t)((data[5] << 8) | data[4]);

  return true;
}

bool configureSensor() {
  // Reset the sensor (reads STAT1 which clears RT bit)
  if (!readRegister(REG_STAT1, &stat1)) return false;
  delay(50);
  // - X, Y, Z enabled (bits 4-6 = 1)
  // - Continuous mode 100Hz (mode = 6)
  uint8_t ctrl1 = 0x70 | 0x0C; 
  if (!writeRegister(REG_CTRL1, ctrl1)) return false;

  delay(30); // Allow time for configuration to take effect
  return true;
}

void printDiagnostics() {
  Serial.print("STAT1: 0x"); Serial.print(stat1, HEX);
  Serial.print(" STAT2: 0x"); Serial.print(stat2, HEX);
  Serial.print(" X: "); Serial.print(x_val);
  Serial.print(" Y: "); Serial.print(y_val);
  Serial.print(" Z: "); Serial.println(z_val);
  
  if (stat2 & 0x01) Serial.println("X overflow!");
  if (stat2 & 0x02) Serial.println("Y overflow!");
  if (stat2 & 0x04) Serial.println("Z overflow!");
  if (stat2 & 0x08) Serial.println("Data overrun!");
}

void setup() {
  //Serial.begin(2000000);
  Serial.begin(115200);
  while (!Serial);
  
  Wire.begin();
  Wire.setClock(freq);

  Serial.println("Initializing MLX90394...");
  
  if (!checkSensorConnection()) {
    Serial.println("Sensor not found!");
    while(1);
  }

  if (!configureSensor()) {
    Serial.println("Configuration failed!");
    while(1);
  }

  Serial.println("Sensor ready");
  Serial.println("X\tY\tZ\tSTAT1\tSTAT2");
}

void loop() {
  static uint32_t lastSampleTime = 0;
  const uint32_t sampleInterval = 3300; //~300hz
  //const uint32_t sampleInterval = 1428; // microseconds for 700Hz
  uint32_t currentTime = micros();
  // Handle micros() overflow (after ~70 minutes)
  if (currentTime < lastSampleTime) {
    lastSampleTime = currentTime;
  }
  // Check if it's time to sample
  if (currentTime - lastSampleTime >= sampleInterval) {
    lastSampleTime = currentTime;
    
    if (readSensorData()) {
      // For fastest output, use minimal serial formatting
      Serial.print(x_val); Serial.print(',');
      Serial.print(y_val); Serial.print(',');
      Serial.println(z_val);
      
      /* even faster option?
      byte buffer[6];
      buffer[0] = x_val & 0xFF;
      buffer[1] = (x_val >> 8) & 0xFF;
      buffer[2] = y_val & 0xFF;
      buffer[3] = (y_val >> 8) & 0xFF;
      buffer[4] = z_val & 0xFF;
      buffer[5] = (z_val >> 8) & 0xFF;
      Serial.write(buffer, 6);
      */
      float B_raw[3] = {avg_x, avg_y, avg_z};
      float B_norm[3];
      float F[3];
      normalize_input(B_raw, B_norm);
      // predict_force(B_norm, F);
      predict_force(B_norm, F); 
    }
  }
  
  // Add any non-time-critical code here
}

void normalize_input(const float raw[3], float norm[3]) {
  for (int i = 0; i < 3; i++) {
    norm[i] = (raw[i] - mu[i]) / sigma[i];
  }
}

void relu(float* vec, int size) {
  for (int i = 0; i < size; i++) {
    if (vec[i] < 0) vec[i] = 0;
  }
}

void predict_force(const float B[3], float F[3]) {
  int N1 = 8, N2 = 8;
  float z1[N1] = {0};
  float z2[8] = {0};

  // Layer 1
  for (int i = 0; i < N1; i++) {
    for (int j = 0; j < 3; j++) {
      z1[i] += W1[i][j] * B[j];
    }
    z1[i] += b1[i];
    if (z1[i] < 0) z1[i] = 0;  // ReLU
  }

  // Layer 2
  for (int i = 0; i < N1; i++) {
    for (int j = 0; j < 8; j++) {
      z2[i] += W2[i][j] * z1[j];
    }
    z2[i] += b2[i];
    if (z2[i] < 0) z2[i] = 0;  // ReLU
  }

  // Output layer
  for (int i = 0; i < 3; i++) {
    F[i] = 0;
    for (int j = 0; j < 8; j++) {
      F[i] += W3[i][j] * z2[j];
    }
    F[i] += b3[i];  // No activation
  }
}

float smooth() { /* function smooth */
  // subtract the last reading:
  total_x = total_x - read_x[readIndex];
  total_y = total_y - read_y[readIndex];
  total_z = total_z - read_z[readIndex];
  // read the sensor:
  read_x[readIndex] = x_val;
  read_y[readIndex] = y_val;
  read_z[readIndex] = z_val;
  // add value to total:
  total_x = total_x + read_x[readIndex];
  total_y = total_y + read_y[readIndex];
  total_z = total_z + read_z[readIndex];
  // handle index
  readIndex = readIndex + 1;
  if (readIndex >= numReadings) {
    readIndex = 0;
  }
  // calculate the average:
  avg_x = total_x / numReadings;
  avg_y = total_y / numReadings;
  avg_z = total_z / numReadings;
}
