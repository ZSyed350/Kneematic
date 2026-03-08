#include <Arduino.h>

#define PWM_PIN 3
#define FULL_REV 360.0f

float pos2duty(float pos);
float get_position();
void drive_PWM(float d);
void sanity_check();