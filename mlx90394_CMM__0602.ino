#include <Wire.h>

// Sensor configuration
const byte I2CAddress = 0x60; // Default address for MLX90394
const long freq = 700000;     // I2C Frequency in Hz

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

// Data variables
int16_t x_val = 0, y_val = 0, z_val = 0;
uint8_t stat1 = 0, stat2 = 0;

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
  const uint32_t sampleInterval = 3333;
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
      /*
      Serial.print(x_val); Serial.print(',');
      Serial.print(y_val); Serial.print(',');
      Serial.println(z_val);
      even faster option?
      */
      uint8_t packet[8];
      packet[0] = 0xAA;  // Header byte 1
      packet[1] = 0x55;  // Header byte 2
      packet[2] = x_val & 0xFF;
      packet[3] = (x_val >> 8) & 0xFF;
      packet[4] = y_val & 0xFF;
      packet[5] = (y_val >> 8) & 0xFF;
      packet[6] = z_val & 0xFF;
      packet[7] = (x_val ^ y_val ^ z_val) & 0xFF; // Simple checksum
      
      Serial.write(packet, 8);
    }
    /*
    else {
      // Only print errors occasionally to avoid slowing down
      static uint8_t errorCount = 0;
      if (++errorCount >= 10) {
        errorCount = 0;
        Serial.println("E"); // Single char error indicator
      }
    */  
    }
  
  
  // Add any non-time-critical code here
}