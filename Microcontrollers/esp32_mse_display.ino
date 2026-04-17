// ============================================================
//  esp32_mse_display.ino  —  Receptor 2 standalone (3 MCUs)
// ============================================================
//  Versión standalone del Receptor 2 para cuando se usan tres
//  microcontroladores físicamente separados.
//
//  En la configuración recomendada de 2 ESP32, esta funcionalidad
//  está integrada en esp32_ifft_audio.ino (taskMetrics, Core 1).
//  Use este archivo solo si dispone de un tercer ESP32.
//
//  Funcionalidad:
//    - Recibe paquetes del Transmisor via UART2 (GPIO 16=RX)
//    - Reconstruye la señal con IFFT (simetría hermítica)
//    - Calcula MSE, energía preservada y SNR (dB)
//    - Muestra resultados en LCD 20×4 I2C
//
//  Nota de conexión para 3 MCUs:
//    El TX2 del Transmisor puede alimentar ambos receptores en
//    paralelo (mismo cable, ambos RX2 conectados con resistencias
//    de 100 Ω en serie para protección de bus).
//    ESP32 TX (3.3V lógica) puede manejar dos entradas RX sin
//    problemas a este baud rate.
//
//  Protocolo recibido (UART2, 921600 baud):
//    [0xAA][0x55] N:u16 original:N×i16 K:u16 K×(idx:u16,re:f32,im:f32) crc:u8
//
//  Conexión hardware:
//    GPIO 16 (RX2) ◄── GPIO 17 (TX2) del Transmisor
//    GND           ◄── GND           del Transmisor
//    GPIO 21 (SDA) ──► LCD I2C SDA
//    GPIO 22 (SCL) ──► LCD I2C SCL
// ============================================================

#include <Arduino.h>
#include <arduinoFFT.h>
#include <LiquidCrystal_I2C.h>

// ── Parámetros (deben coincidir con el Transmisor) ──────────
#define N            256
#define UART2_RX_PIN 16
#define UART2_TX_PIN 17
#define UART2_BAUD   921600

// ── Cabeceras de protocolo ──────────────────────────────────
#define ESP_HDR_A  0xAA
#define ESP_HDR_B  0x55

// ── Buffers globales ────────────────────────────────────────
static double     vReal[N];
static double     vImag[N];
static arduinoFFT FFT;

static int16_t  originalSamples[N];
static uint16_t coeffIndices[N / 2 + 1];
static float    coeffRe[N / 2 + 1];
static float    coeffIm[N / 2 + 1];
static uint16_t numCoeffs;

// ── LCD 20×4 I2C ───────────────────────────────────────────
static LiquidCrystal_I2C lcd(0x27, 20, 4);

// ────────────────────────────────────────────────────────────
//  Leer exactamente `len` bytes de Serial2 (bloqueante)
// ────────────────────────────────────────────────────────────
static uint8_t rxCrc;

static void readByte(uint8_t& b) {
    while (!Serial2.available()) {}
    b = Serial2.read();
    rxCrc ^= b;
}

static void readBuf(void* dst, size_t len) {
    uint8_t* p = reinterpret_cast<uint8_t*>(dst);
    for (size_t i = 0; i < len; i++) readByte(p[i]);
}

// ────────────────────────────────────────────────────────────
//  Reconstruir espectro completo con simetría hermítica
// ────────────────────────────────────────────────────────────
static void reconstructSpectrum() {
    memset(vReal, 0, sizeof(vReal));
    memset(vImag, 0, sizeof(vImag));

    for (uint16_t k = 0; k < numCoeffs; k++) {
        uint16_t idx = coeffIndices[k];
        vReal[idx] = static_cast<double>(coeffRe[k]);
        vImag[idx] = static_cast<double>(coeffIm[k]);

        // Conjugado simétrico (señal real)
        if (idx > 0 && idx < N / 2) {
            vReal[N - idx] =  static_cast<double>(coeffRe[k]);
            vImag[N - idx] = -static_cast<double>(coeffIm[k]);
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Setup
// ────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial2.begin(UART2_BAUD, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);

    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0);
    lcd.print("Receptor 2 (MSE)");
    lcd.setCursor(0, 1);
    lcd.print("Esperando datos...");
}

// ────────────────────────────────────────────────────────────
//  Loop principal
// ────────────────────────────────────────────────────────────
void loop() {
    // 1. Sincronizar con cabecera [0xAA][0x55]
    for (;;) {
        while (!Serial2.available()) {}
        if (Serial2.read() != ESP_HDR_A) continue;
        while (!Serial2.available()) {}
        if (Serial2.peek() == ESP_HDR_B) { Serial2.read(); break; }
    }

    rxCrc = 0;

    // 2. Leer tamaño de bloque N
    uint16_t blockN = 0;
    readBuf(&blockN, sizeof(blockN));
    if (blockN != N) return;  // descarta paquete inesperado

    // 3. Muestras originales
    readBuf(originalSamples, N * sizeof(int16_t));

    // 4. K coeficientes
    readBuf(&numCoeffs, sizeof(numCoeffs));
    if (numCoeffs > N / 2 + 1) return;

    // 5. Coeficientes
    for (uint16_t k = 0; k < numCoeffs; k++) {
        readBuf(&coeffIndices[k], sizeof(uint16_t));
        readBuf(&coeffRe[k],      sizeof(float));
        readBuf(&coeffIm[k],      sizeof(float));
    }

    // 6. Validar CRC
    uint8_t calcCrc = rxCrc;
    uint8_t rxd;
    while (!Serial2.available()) {}
    rxd = Serial2.read();
    if (rxd != calcCrc) {
        Serial.println("[MSE] CRC error, descartando paquete");
        return;
    }

    // 7. Reconstruir espectro e IFFT
    reconstructSpectrum();
    FFT.Compute(vReal, vImag, N, FFT_REVERSE);

    // 8. Calcular métricas
    double mse         = 0.0;
    double energyOrig  = 0.0;
    double energyRecon = 0.0;
    double errorEnergy = 0.0;

    for (int i = 0; i < N; i++) {
        double orig  = static_cast<double>(originalSamples[i]);
        double recon = vReal[i] / N;
        double err   = orig - recon;

        mse         += err * err;
        energyOrig  += orig  * orig;
        energyRecon += recon * recon;
        errorEnergy += err   * err;
    }
    mse /= N;

    double energyPct = (energyOrig > 0.0)
                       ? (energyRecon / energyOrig) * 100.0
                       : 0.0;

    double snr = (errorEnergy > 1e-9)
                 ? 10.0 * log10(energyOrig / errorEnergy)
                 : 99.9;

    // 9. Debug serial
    Serial.print("[MSE] MSE="); Serial.print(mse, 2);
    Serial.print("  En=");      Serial.print(energyPct, 1); Serial.print("%");
    Serial.print("  SNR=");     Serial.print(snr, 1);       Serial.print("dB");
    Serial.print("  K=");       Serial.println(numCoeffs);

    // 10. Mostrar en LCD
    lcd.clear();

    lcd.setCursor(0, 0);
    lcd.print("MSE: ");
    lcd.print(mse, 2);

    lcd.setCursor(0, 1);
    lcd.print("Energia: ");
    lcd.print(energyPct, 1);
    lcd.print("%");

    lcd.setCursor(0, 2);
    lcd.print("SNR: ");
    lcd.print(snr, 1);
    lcd.print(" dB");

    lcd.setCursor(0, 3);
    lcd.print("K=");
    lcd.print(numCoeffs);
    lcd.print("/");
    lcd.print(N / 2 + 1);
    lcd.print(" bins");
}
