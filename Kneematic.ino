#include "knee_position.h"
#include "em_braking.h"
#include "data_out.h"

void setup() {
    Serial.begin(115200); 
    Serial.setTimeout(1);

    pinMode(PWM_PIN, OUTPUT);
    pinMode(POT_PIN, INPUT);
}

void _main() {
    float pos = get_position();
    drive_PWM(pos2duty(pos));
    write2csv();
    delay(20);
}

void loop() {
    // _main();
    // EM_sanity_check();
    EM_calibrate(100.0);
    // float angle = getAngle();
    // Serial.println(angle);
}