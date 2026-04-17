#include <arduinoFFT.h>
#include <LiquidCrystal_I2C.h>

#define N 64

arduinoFFT FFT = arduinoFFT();
LiquidCrystal_I2C lcd(0x27, 20, 4);

double vReal[N];
double vImag[N];
double reconstructed[N];

void setup() {
  Serial.begin(115200);
  lcd.init();
  lcd.backlight();
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');

    parseData(data);

    // IFFT
    FFT.Compute(vReal, vImag, N, FFT_REVERSE);

    for (int i = 0; i < N; i++) {
      reconstructed[i] = vReal[i] / N;
    }

    // Calcular MSE (vs señal original aproximada)
    double mse = 0;

    for (int i = 0; i < N; i++) {
      double original = 0; // aquí podrías reconstruir mejor si envías original
      mse += pow(original - reconstructed[i], 2);
    }

    mse /= N;

    // Mostrar en LCD
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("MSE:");
    lcd.setCursor(0, 1);
    lcd.print(mse);

    delay(500);
  }
}

void parseData(String data) {
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