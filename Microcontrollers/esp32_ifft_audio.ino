#include <arduinoFFT.h>

#define N 64
#define SPEAKER_PIN 25

arduinoFFT FFT = arduinoFFT();

double vReal[N];
double vImag[N];

void setup() {
  Serial.begin(115200);
  ledcAttachPin(SPEAKER_PIN, 0);
  ledcSetup(0, 8000, 8); // PWM
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');

    parseData(data);

    // IFFT
    FFT.Compute(vReal, vImag, N, FFT_REVERSE);

    // Normalizar
    for (int i = 0; i < N; i++) {
      vReal[i] /= N;
    }

    // Reproducir señal
    for (int i = 0; i < N; i++) {
      int value = (int)(vReal[i] + 128);
      value = constrain(value, 0, 255);
      ledcWrite(0, value);
      delayMicroseconds(1000);
    }
  }
}

void parseData(String data) {
  int index = 0;
  int start = 0;

  for (int i = 0; i < N; i++) {
    int comma = data.indexOf(',', start);
    int semicolon = data.indexOf(';', start);

    if (semicolon == -1) semicolon = data.length();

    vReal[i] = data.substring(start, comma).toFloat();
    vImag[i] = data.substring(comma + 1, semicolon).toFloat();

    start = semicolon + 1;
  }
}