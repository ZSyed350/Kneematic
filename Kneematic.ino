#include "knee_position.h"
#include "em_braking.h"
#include "data_out.h"

int dir = -1;
int prev_pos = 0;

void setup() {
    Serial.begin(115200); 
    Serial.setTimeout(1);

    pinMode(PWM_PIN, OUTPUT);
    pinMode(POT_PIN, INPUT);
}

// 1 = flexion, -1 = extension
float _main(int dir) {
    float pos = getAngle();
    drive_PWM(pos2duty(pos, dir));
    // write2csv();
    delay(20);

    return pos;
}

void loop() {
    float pos = _main(prev_pos);
    if (pos > prev_pos) {
        dir = -1;
    }
    else {
        dir = 0;
    }
    prev_pos = pos;
}