#include "knee_position.h"
#include "em_braking.h"
#include "data_out.h"

void setup() {
    pinMode(PWM_PIN, OUTPUT);
}

void main() {
    float pos = get_position();
    drive_pwm(pos2duty(pos));
    write2csv();
    delay(20);
}

void loop() {
    // main();
    sanity_check();
}