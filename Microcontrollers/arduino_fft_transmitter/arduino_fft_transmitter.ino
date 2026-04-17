#include <arduinoFFT.h>

#define N 64
#define SAMPLING_FREQUENCY 4000

arduinoFFT FFT = arduinoFFT();

double vReal[N];
double vImag[N];

const int micPin = A0;

int ADC_OFFSET = 512; // se recalibra en setup

void setup() {
  Serial.begin(115200);

  // 🔧 Calibración automática del offset
  long sum = 0;
  for (int i = 0; i < 200; i++) {
    sum += analogRead(micPin);
    delay(5);
  }
  ADC_OFFSET = sum / 200;

  Serial.print("Offset calibrado: ");
  Serial.println(ADC_OFFSET);
}

void loop() {
  // 🔊 Muestreo de audio
  for (int i = 0; i < N; i++) {
    int raw = analogRead(micPin);

    // Centrar señal en 0
    vReal[i] = raw - ADC_OFFSET;
    vImag[i] = 0;

    delayMicroseconds(1000000 / SAMPLING_FREQUENCY);
  }

  // 🧠 FFT
  FFT.Windowing(vReal, N, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.Compute(vReal, vImag, N, FFT_FORWARD);

  // 🔥 Compresión basada en magnitud
  double maxMagnitude = 0;

  for (int i = 0; i < N; i++) {
    double mag = sqrt(vReal[i]*vReal[i] + vImag[i]*vImag[i]);
    if (mag > maxMagnitude) maxMagnitude = mag;
  }

  // Umbral adaptativo (20% del máximo)
  double threshold = 0.2 * maxMagnitude;

  for (int i = 0; i < N; i++) {
    double mag = sqrt(vReal[i]*vReal[i] + vImag[i]*vImag[i]);

    if (mag < threshold) {
      vReal[i] = 0;
      vImag[i] = 0;
    }
  }

  // 📡 Envío serial (real,imag)
  for (int i = 0; i < N; i++) {
    Serial.print(vReal[i]);
    Serial.print(",");
    Serial.print(vImag[i]);

    if (i < N - 1) Serial.print(";");
  }
  Serial.println();

  delay(50);
}