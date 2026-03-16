#include "em_braking.h"
#include "knee_position.h"

float pos2duty(float pos) {
  if (pos <= 0.0f) { return 0.0f; }
  if (pos >= FULL_REV) { return 1.0f; }
  return (pos / FULL_REV);
}

void drive_PWM(float d) {
  int duty = (int)(d * 255.0f);
  analogWrite(PWM_PIN, duty);
}

void EM_calibrate(float d) {
  float angle = getAngle();

  // Before experiment
  if (angle > 0) {
    Serial.println("Angle greater than 0. Move member to starting position.");
  }
  while (angle > 0) {
    angle = getAngle(); 
  }

  // Member at top position (angles are such that 0.0 is a wider range)
  Serial.println("Member at start position.");
  while (angle == 0.0) {
    angle = getAngle(); 
  }

  // Start EM
  Serial.print("Starting EM at duty cycle ");
  Serial.println(d);
  drive_PWM(d);
  Serial.println("=== START DATA ===");

  // End position is a bit less than 200
  // Manually push to 200 to end test
  while (angle < 200) {
    angle = getAngle();             // read angle
    unsigned long timestamp = millis();   // milliseconds since start

    Serial.print(timestamp);
    Serial.print(", ");
    Serial.println(angle, 4);   // print with 4 decimal places
  }
  Serial.println("=== END DATA ===");
  
  drive_PWM(0.0);
  while(true);
}

void EM_sanity_check() {
  int d = 0;
  int dir = 1;      // 1 = up, -1 = down

  for (int cycle = 0; cycle < 4; cycle++) {
    for (int d = 0; d <= 255; d++) {
      drive_PWM(d);
      delay(30);
    }
    for (int d = 255; d > 0; d--) {
      drive_PWM(d);
      delay(30);
    }
    drive_PWM(0);
    delay(30);
  }
}
