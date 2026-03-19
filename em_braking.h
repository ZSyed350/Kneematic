#include <Arduino.h>

#define PWM_PIN 3
#define FULL_ROM 140.0f

float pos2duty(float pos, int dir);
float get_position();
void drive_PWM(float d);
void EM_calibrate(float d);
void EM_sanity_check();