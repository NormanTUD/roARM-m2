#include <SCServo.h>

SMS_STS sms_sts;

#define S_RXD 19
#define S_TXD 18

long baudrates[] = {1000000, 500000, 250000, 115200, 57600, 38400};
int num_bauds = 6;

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("=== ST3215 Multi-Baudrate Scan ===");
  
  int found_id = -1;
  long found_baud = 0;
  
  for (int b = 0; b < num_bauds; b++) {
    Serial.printf("\nTrying baudrate %ld...\n", baudrates[b]);
    Serial1.begin(baudrates[b], SERIAL_8N1, S_RXD, S_TXD);
    sms_sts.pSerial = &Serial1;
    delay(100);
    
    for (int id = 0; id <= 10; id++) {
      int pos = sms_sts.ReadPos(id);
      if (pos != -1) {
        Serial.printf("  FOUND! Servo at ID %d, baud %ld (pos=%d)\n", id, baudrates[b], pos);
        found_id = id;
        found_baud = baudrates[b];
      }
      delay(20);
    }
    Serial1.end();
  }
  
  if (found_id == -1) {
    Serial.println("\n!!! NO SERVO FOUND AT ANY BAUDRATE !!!");
    Serial.println("Check: Is 12V power connected?");
    Serial.println("Check: Is servo cable plugged in correctly?");
    return;
  }
  
  if (found_id == 4 && found_baud == 1000000) {
    Serial.println("Servo already has ID 4 at 1Mbaud - nothing to do!");
    return;
  }
  
  // Reconnect at found baudrate
  Serial1.begin(found_baud, SERIAL_8N1, S_RXD, S_TXD);
  sms_sts.pSerial = &Serial1;
  delay(100);
  
  Serial.printf("\nChanging ID from %d to 4...\n", found_id);
  sms_sts.unLockEprom(found_id);
  delay(100);
  sms_sts.writeByte(found_id, SMS_STS_ID, 4);
  delay(100);
  sms_sts.LockEprom(4);
  delay(100);
  
  int verify_pos = sms_sts.ReadPos(4);
  if (verify_pos != -1) {
    Serial.printf("SUCCESS! Servo now responds at ID 4 (pos=%d)\n", verify_pos);
  } else {
    Serial.println("WARNING: Verify failed - try power cycling");
  }
}

void loop() {
  delay(1000);
}
