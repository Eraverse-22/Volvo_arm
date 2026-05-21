#include <ESP32Servo.h>

Servo servo[6];
const int PINS[6] = {4, 16, 13, 14, 27, 26};

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 6; i++) {
    servo[i].attach(PINS[i], 500, 2500);
    servo[i].write(90);
  }
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("A")) {
      line = line.substring(1);
      int angles[6];
      int idx = 0;
      char buf[64];
      line.toCharArray(buf, sizeof(buf));
      char* token = strtok(buf, ",");
      while (token != NULL && idx < 6) {
        angles[idx] = constrain(atoi(token), 0, 180);
        token = strtok(NULL, ",");
        idx++;
      }
      if (idx == 6) {
        for (int i = 0; i < 6; i++) {
          servo[i].write(angles[i]);
        }
        Serial.println("OK");
      } else {
        Serial.println("ERR:bad_count");
      }
    }
    else if (line == "HOME") {
      for (int i = 0; i < 6; i++) servo[i].write(90);
      Serial.println("OK:home");
    }
    else if (line == "PING") {
      Serial.println("PONG");
    }
  }
}
