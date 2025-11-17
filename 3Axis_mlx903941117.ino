//70% deepseek code here, I'm gonna LLM the shit out of it
#include <Wire.h>
#include "fcnn_weights.h"


// Sensor configuration
const byte I2CAddress1 = 0x10;  // First sensor address
const byte I2CAddress2 = 0x68;  // Second sensor address
const byte I2CAddress3 = 0x60;  // Third sensor address (adjust as needed)
const long freq = 2000000;      // I2C Frequency in Hz (2MHz)

// Calibration parameters for each sensor

//const float mu1[3] = {208.812205008607, -45.5994225109667, 14.5227941584763};
//const float sigma1[3] = {202.115205105661, 93.0846055668491, 67.7296408356276};
//const int base_x1 = 1715, base_y1 = 2194, base_z1 = 106;

//const float mu2[3] = {208.812205008607, -45.5994225109667, 14.5227941584763};
//const float sigma2[3] = {202.115205105661, 93.0846055668491, 67.7296408356276};
//const int base_x2 = 1715, base_y2 = 2194, base_z2 = 106;

//const float mu3[3] = {208.812205008607, -45.5994225109667, 14.5227941584763}; // Adjust for sensor 3
//const float sigma3[3] = {202.115205105661, 93.0846055668491, 67.7296408356276}; // Adjust for sensor 3
//const int base_x3 = 1715, base_y3 = 2194, base_z3 = 106; // Adjust for sensor 3

const float mu1[3] = {15, 15, 15};
const float sigma1[3] = {15, 15, 15};
const int base_x1 = 0, base_y1 = 0, base_z1 = 0;
const float mu2[3] = {15, 15, 15};
const float sigma2[3] = {15, 15, 15};
const int base_x2 = 0, base_y2 = 0, base_z2 = 0;
const float mu3[3] = {15, 15, 15};
const float sigma3[3] = {15, 15, 15};
const int base_x3 = 0, base_y3 = 0, base_z3 = 0;

const float delay_millis = 15; // in milliseconds
const int numReadings = 10;    // Moving average window size

// Register addresses (same for all sensors)
#define REG_STAT1 0x00
#define REG_X_L 0x01
#define REG_X_H 0x02
#define REG_Y_L 0x03
#define REG_Y_H 0x04
#define REG_Z_L 0x05
#define REG_Z_H 0x06
#define REG_T_L 0x07   // ADDED: Temperature Low Byte
#define REG_T_H 0x08   // ADDED: Temperature High Byte
#define REG_STAT2 0x07 // This is incorrect in original, STAT2 is 0x09. But T_L is 0x07.
                       // The read sequence is X,Y,Z,T,STAT2. We'll read from 0x01 to 0x09.
                       // Let's redefine STAT2 correctly.
#undef REG_STAT2
#define REG_STAT2 0x09 // ADDED: Correct STAT2 Address
#define REG_CTRL1 0x0E
#define REG_CTRL3 0x14 // ADDED: From datasheet image
#define REG_CTRL4 0x15 // ADDED: From datasheet image

// Data structures for each sensor
struct SensorData {
  int16_t x_val = 0, y_val = 0, z_val = 0, temp_val = 0; // ADDED: temp_val
  uint8_t stat1 = 0, stat2 = 0;
  float avg_x = 0, avg_y = 0, avg_z = 0, avg_temp = 0;  // ADDED: avg_temp
  float total_x = 0, total_y = 0, total_z = 0, total_temp = 0; // ADDED: total_temp
  int readIndex = 0;
  int read_x[10] = {0};  // Matches numReadings size
  int read_y[10] = {0};
  int read_z[10] = {0};
  int read_temp[10] = {0}; // ADDED: read_temp
};

SensorData sensor1, sensor2, sensor3;

bool checkSensorConnection(byte address) {
  Wire.beginTransmission(address);
  return (Wire.endTransmission() == 0);
}

bool readRegister(byte address, uint8_t reg, uint8_t* value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  
  Wire.requestFrom(address, (uint8_t)1);
  if (Wire.available()) {
    *value = Wire.read();
    return true;
  }
  return false;
}

bool writeRegister(byte address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  return (Wire.endTransmission() == 0);
}

bool dataReady(byte address, uint8_t* stat1) {
  return readRegister(address, REG_STAT1, stat1) && (*stat1 & 0x01);
}

bool readSensorData(byte address, SensorData* data) {
  // Check if data is ready
  if (!dataReady(address, &data->stat1)) {
    return false;
  }

  // Read measurement data (X, Y, Z, and Temp)
  Wire.beginTransmission(address);
  Wire.write(REG_X_L); // Start reading from X_L (0x01)
  if (Wire.endTransmission(false) != 0) return false;
  
  // We read 8 bytes for X,Y,Z,T (0x01-0x08) + 1 byte for STAT2 (0x09)
  uint8_t sensorData[9]; // CHANGED: 9 bytes total
  Wire.requestFrom(address, (uint8_t)9); // CHANGED: 9 bytes total
  for (int i = 0; i < 9; i++) { // CHANGED: 9
    if (!Wire.available()) return false;
    sensorData[i] = Wire.read();
  }

  // STAT2 is the 9th byte (index 8)
  data->stat2 = sensorData[8]; // CHANGED: Get STAT2 from the block read

  // Convert data to 16-bit values
  data->x_val = (int16_t)((sensorData[1] << 8) | sensorData[0]);
  data->y_val = (int16_t)((sensorData[3] << 8) | sensorData[2]);
  data->z_val = (int16_t)((sensorData[5] << 8) | sensorData[4]);
  data->temp_val = (int16_t)((sensorData[7] << 8) | sensorData[6]); // ADDED: temp

  return true;
}

bool configureSensor(byte address) {
  // Reset the sensor
  uint8_t stat1;
  if (!readRegister(address, REG_STAT1, &stat1)) return false;
  delay(60);
  
  // Configuration
  
  // ADDED: Configure CTRL3 (0x14)
  // Using default values from datasheet (RW-1 or RW-0)
  // OSR_HALL=1, OSR_TEMP=1, DIG_FILT_HALL_XY=0, DIG_FILT_TEMP=1
  uint8_t ctrl3 = 0xC1; // 1100 0001
  if (!writeRegister(address, REG_CTRL3, ctrl3)) return false;
  delay(1);

  // ADDED: Configure CTRL4 (0x15)
  // Set T_EN (Bit 5) to 1. Set reserved bits as per datasheet.
  // Bit 7: CTRL4_7 (1)
  // Bit 6: Reserved (0)
  // Bit 5: T_EN (1) <-- This enables temperature
  // Bit 4: CTRL4_4 (1)
  // Bit 3: DRDY_EN (0)
  // Bit 2-0: DIG_FILT_HALL_Z (111)
  uint8_t ctrl4 = 0xB7; // 1011 0111
  if (!writeRegister(address, REG_CTRL4, ctrl4)) return false;
  delay(1);

  // CTRL1: X, Y, Z enabled, Continuous mode 100Hz
  uint8_t ctrl1 = 0x70 | 0x0C; 
  if (!writeRegister(address, REG_CTRL1, ctrl1)) return false;

  delay(40);
  return true;
}

void smooth(SensorData* data) {
  // subtract the last reading:
  data->total_x -= data->read_x[data->readIndex];
  data->total_y -= data->read_y[data->readIndex];
  data->total_z -= data->read_z[data->readIndex];
  data->total_temp -= data->read_temp[data->readIndex]; // ADDED
  
  // read the sensor (values already updated in readSensorData):
  data->read_x[data->readIndex] = data->x_val;
  data->read_y[data->readIndex] = data->y_val;
  data->read_z[data->readIndex] = data->z_val;
  data->read_temp[data->readIndex] = data->temp_val; // ADDED
  
  // add value to total:
  data->total_x += data->read_x[data->readIndex];
  data->total_y += data->read_y[data->readIndex];
  data->total_z += data->read_z[data->readIndex];
  data->total_temp += data->read_temp[data->readIndex]; // ADDED
  
  // handle index
  data->readIndex = (data->readIndex + 1) % numReadings;
  
  // calculate the average:
  data->avg_x = data->total_x / numReadings;
  data->avg_y = data->total_y / numReadings;
  data->avg_z = data->total_z / numReadings;
  data->avg_temp = data->total_temp / numReadings; // ADDED
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  Wire.begin();
  Wire.setClock(freq);

  Serial.println("Initializing MLX90394 sensors...");
  
  // Initialize sensor 1
  if (!checkSensorConnection(I2CAddress1)) {
    Serial.println("Sensor 1 not found!");
    while(1);
  }
  if (!configureSensor(I2CAddress1)) {
    Serial.println("Sensor 1 configuration failed!");
    while(1);
  }

  // Initialize sensor 2
  if (!checkSensorConnection(I2CAddress2)) {
    Serial.println("Sensor 2 not found!");
    while(1);
  }
  if (!configureSensor(I2CAddress2)) {
    Serial.println("Sensor 2 configuration failed!");
    while(1);
  }

  // Initialize sensor 3
  if (!checkSensorConnection(I2CAddress3)) {
    Serial.println("Sensor 3 not found!");
    while(1);
  }
  if (!configureSensor(I2CAddress3)) {
    Serial.println("Sensor 3 configuration failed!");
    while(1);
  }

  Serial.println("All sensors ready");
  // CHANGED: Added Temp columns
  Serial.println("X1\tY1\tZ1\tT1\tX2\tY2\tZ2\tT2\tX3\tY3\tZ3\tT3");
}

void loop() {
  static uint32_t lastSampleTime = 0;
  const uint32_t sampleInterval = 3300; // ~300Hz
  
  uint32_t currentTime = micros();
  if (currentTime < lastSampleTime) {
    lastSampleTime = currentTime; // Handle overflow
  }
  
  if (currentTime - lastSampleTime >= sampleInterval) {
    lastSampleTime = currentTime;
    
    // Read sensor 1
    if (readSensorData(I2CAddress1, &sensor1)) {
      smooth(&sensor1);
      float B_raw1[3] = {sensor1.avg_x, sensor1.avg_y, sensor1.avg_z};
      float B_norm1[3];
      float F1[3];
      normalize_input(B_raw1, B_norm1, mu1, sigma1);
      predict_force(B_norm1, F1);
    }
    
    // Read sensor 2
    if (readSensorData(I2CAddress2, &sensor2)) {
      smooth(&sensor2);
      float B_raw2[3] = {sensor2.avg_x, sensor2.avg_y, sensor2.avg_z};
      float B_norm2[3];
      float F2[3];
      normalize_input(B_raw2, B_norm2, mu2, sigma2);
      predict_force(B_norm2, F2);
    }
    
    // Read sensor 3
    if (readSensorData(I2CAddress3, &sensor3)) {
      smooth(&sensor3);
      float B_raw3[3] = {sensor3.avg_x, sensor3.avg_y, sensor3.avg_z};
      float B_norm3[3];
      float F3[3];
      normalize_input(B_raw3, B_norm3, mu3, sigma3);
      predict_force(B_norm3, F3);
    }
    
    // Output data for all three sensors
    Serial.print(sensor1.x_val); Serial.print(',');
    Serial.print(sensor1.y_val); Serial.print(',');
    Serial.print(sensor1.z_val); Serial.print(',');
    Serial.print(sensor1.temp_val); Serial.print(','); // ADDED
    Serial.print(sensor2.x_val); Serial.print(',');
    Serial.print(sensor2.y_val); Serial.print(',');
    Serial.print(sensor2.z_val); Serial.print(',');
    Serial.print(sensor2.temp_val); Serial.print(','); // ADDED
    Serial.print(sensor3.x_val); Serial.print(',');
    Serial.print(sensor3.y_val); Serial.print(',');
    Serial.print(sensor3.z_val); Serial.print(',');
    Serial.println(sensor3.temp_val); // CHANGED
  }
}
