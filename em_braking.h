#include <Arduino.h>

#define PWM_PIN 9

float pos2duty(float pos);
void drive_PWM(float d);
void sanity_check();