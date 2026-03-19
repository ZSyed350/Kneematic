#include "knee_position.h"
#include "em_braking.h"

EM em;
void setup() {
    Serial.begin(115200); 
    Serial.setTimeout(1);

    pinMode(PWM_PIN, OUTPUT);
    pinMode(POT_PIN, INPUT);
}

void loop() {
    float pos = em.EM_main();  // save pos for write out
}