#include "em_braking.h"

// calibration goes here
float pos2duty(float pos) {
  if (pos <= 0.0f) { return 0.0f; }
  if (pos >= FULL_REV) { return 1.0f; }
  return (pos / FULL_REV);
}

void drive_PWM(float d) {
  int duty = (int)(d * 255.0f);
  analogWrite(PWM_PIN, duty);
}

void sanity_check() {
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
