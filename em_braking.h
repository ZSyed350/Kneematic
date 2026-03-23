#ifndef EM_BRAKING_H
#define EM_BRAKING_H

#include <Arduino.h>

#define PWM_PIN 3
#define FULL_ROM 140.0f

class EM {
public:
    EM();
    float EM_main();

private:
    int dir;              // 1 = flexion, -1 = extension
    int prev_pos;
    float initial_angle;

    void drive_PWM(float d);
    float pos2duty(float pos);
};

#endif