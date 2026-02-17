/* DEFINES */
#define PWM_PIN 9
#define CH_B 3
#define CH_A 2

/* CONSTANTS */
const float COUNTS_PER_REV = 2048.0;
const float FULL_REV = 360.0;

/* GLOBAL */
volatile long counts = 0;
volatile uint8_t lastA = 0;
float prev_pos = 0.00;

void ISR_A_change() {
  uint8_t a = digitalRead(CH_A);
  uint8_t b = digitalRead(CH_B);

  if (a != lastA) {
    counts += (a == b) ? 1 : -1;
    lastA = a;
  }
}

void setup() {
  pinMode(PWM_PIN, OUTPUT);
  pinMode(CH_A, INPUT);
  pinMode(CH_B, INPUT);

  lastA = digitalRead(CH_A);
  attachInterrupt(digitalPinToInterrupt(CH_A), ISR_A_change, CHANGE);
  Serial.begin(115200);

  Serial.print("Position (degrees): ");
  Serial.println(prev_pos, 2);
}

float read_pos() {
  long c;
  noInterrupts();  // disable interrupts while reading
  c = counts;
  interrupts();

  float angle = (c / COUNTS_PER_REV) * 360.0;
  return angle;
}

float pos2duty(float pos) {
  if (pos <= 0.0f) { return 0.0f; }
  if (pos >= FULL_REV) { return 1.0f; }
  return (pos / FULL_REV);
}

void drive_PWM(float d) {
  int duty = (int)(d * 255.0f);
  analogWrite(PWM_PIN, duty);
}

void sanity_check() {
  int d = 0;
  int dir = 1;      // 1 = up, -1 = down

  for (int cycle = 0; cycle < 4; cycle++) {
    for (int d = 0; d <= 255; d++) {
      drive_PWM(d);
      delay(30);
    }
    for (int d = 255; d > 0; d--) {
      drive_PWM(d);
      delay(30);
    }
    drive_PWM(0);
  }
}

void loop() {
  float pos = read_pos();

  if (pos != prev_pos) {
    Serial.print("Position (degrees): ");
    Serial.println(pos, 2);
    prev_pos = pos;
  }

  drive_PWM(pos2duty(pos));

  delay(20);
}
