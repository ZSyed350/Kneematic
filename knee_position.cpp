#include "knee_position.h"

float getAngle() {
    int potAnalog = analogRead(POT_PIN);
    float potPos = (float)potAnalog / 1023.0f * MAX_ANGLE;
    return potPos;
}