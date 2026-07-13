#include <SCServo.h>

SMS_STS sms_sts;

// RoArm-M2-S verwendet GPIO 18 (TX) und GPIO 19 (RX) für den Servo-Bus
#define S_RXD 19
#define S_TXD 18

void setup() {
  Serial.begin(115200);
  Serial1.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);
  sms_sts.pSerial = &Serial1;
  
  delay(2000);
  Serial.println("=== ST3215 Servo ID Change ===");
  Serial.println("Scanning for servos...");
  
  // Scan: Welche ID hat der Servo gerade?
  int found_id = -1;
  for (int id = 0; id <= 10; id++) {
    int pos = sms_sts.ReadPos(id);
    if (pos != -1) {
      Serial.printf("  Found servo at ID %d (pos=%d)\n", id, pos);
      found_id = id;
    }
    delay(50);
  }
  
  if (found_id == -1) {
    Serial.println("ERROR: No servo found!");
    return;
  }
  
  if (found_id == 4) {
    Serial.println("Servo already has ID 4 - nothing to do!");
    return;
  }
  
  // ID ändern
  Serial.printf("Changing ID from %d to 4...\n", found_id);
  
  sms_sts.unLockEprom(found_id);
  delay(100);
  sms_sts.writeByte(found_id, SMS_STS_ID, 4);
  delay(100);
  sms_sts.LockEprom(4);
  delay(100);
  
  // Verify
  int verify_pos = sms_sts.ReadPos(4);
  if (verify_pos != -1) {
    Serial.printf("SUCCESS! Servo now responds at ID 4 (pos=%d)\n", verify_pos);
  } else {
    Serial.println("WARNING: Verify failed - try power cycling");
  }
  
  Serial.println("\n>>> Flash the original RoArm-M2-S firmware back now! <<<");
}

void loop() {
  delay(1000);
}