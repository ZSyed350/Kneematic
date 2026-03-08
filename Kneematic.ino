// #include "knee_position.h"
#include "em_braking.h"
#include "data_out.h"

void setup() {
    pinMode(PWM_PIN, OUTPUT);
}

void _main() {
    float pos = get_position();
    drive_PWM(pos2duty(pos));
    write2csv();
    delay(20);
}

void loop() {
    // _main();
    sanity_check();
}