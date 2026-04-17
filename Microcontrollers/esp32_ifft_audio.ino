// ============================================================
//  esp32_dual_receptor.ino  —  ESP32 #2: Receptor Combinado
// ============================================================
//  Implementa Receptor 1 y Receptor 2 del proyecto en un solo
//  ESP32 usando FreeRTOS dual-core, tal como permite el enunciado
//  ("Si se realiza en un microcontrolador multihilo, los dos
//  receptores pueden implementarse en un solo dispositivo").
//
//  Arquitectura FreeRTOS:
//    Core 0 — taskAudio   : IFFT → DAC 8-bit → Speaker  (Receptor 1)
//    Core 1 — taskReceive : parsea paquetes UART2 (alta prioridad)
//    Core 1 — taskMetrics : IFFT → MSE/Energía/SNR → LCD (Receptor 2)
//
//  Protocolo recibido desde ESP32 #1 (UART2, 921600 baud):
//    [0xAA][0x55] N:u16 original:N×i16 K:u16 K×(idx:u16,re:f32,im:f32) crc:u8
//
//  Solo se reciben los bins de la mitad positiva del espectro
//  (0..N/2). El receptor reconstruye la simetría hermítica para
//  obtener el espectro completo antes de aplicar la IFFT.
//
//  Conexión hardware:
//    GPIO 16 (RX2) ◄── GPIO 17 (TX2) de ESP32 #1
//    GND           ◄── GND           de ESP32 #1
//    GPIO 25 (DAC1) ──► RC LPF ──► Amplificador ──► Speaker
//    GPIO 21 (SDA) ──► LCD I2C SDA
//    GPIO 22 (SCL) ──► LCD I2C SCL
// ============================================================

#include <Arduino.h>
#include <arduinoFFT.h>
#include <LiquidCrystal_I2C.h>

// ── Parámetros del sistema ──────────────────────────────────
#define N            256     // Debe coincidir con el transmisor
#define SAMPLE_RATE  8000    // Hz

// ── Pines ───────────────────────────────────────────────────
#define SPEAKER_PIN  25      // DAC1 del ESP32 (8-bit, 0–255 → 0–3.3V)
#define UART2_RX_PIN 16
#define UART2_TX_PIN 17
#define UART2_BAUD   921600

// ── Cabeceras de protocolo ──────────────────────────────────
#define ESP_HDR_A  0xAA
#define ESP_HDR_B  0x55

// ── Paquete recibido (heap) ─────────────────────────────────
struct RecvPacket {
    int16_t  original[N];        // Muestras originales (referencia MSE)
    uint16_t numCoeffs;          // K coeficientes recibidos
    uint16_t indices[N / 2 + 1]; // Índices de bins (mitad positiva)
    float    re[N / 2 + 1];      // Parte real
    float    im[N / 2 + 1];      // Parte imaginaria
};

// ── Colas FreeRTOS (punteros a heap) ───────────────────────
static QueueHandle_t audioQueue;
static QueueHandle_t metricsQueue;

// ── LCD 20×4 I2C ───────────────────────────────────────────
static LiquidCrystal_I2C lcd(0x27, 20, 4);

// ────────────────────────────────────────────────────────────
//  Helper: reconstruir espectro completo N bins a partir de
//  los coeficientes de la mitad positiva usando simetría hermítica.
//  Para señal real: X[N-k] = conj(X[k])
// ────────────────────────────────────────────────────────────
static void reconstructSpectrum(const RecvPacket* pkt,
                                double* re, double* im) {
    memset(re, 0, N * sizeof(double));
    memset(im, 0, N * sizeof(double));

    for (uint16_t k = 0; k < pkt->numCoeffs; k++) {
        uint16_t idx = pkt->indices[k];
        re[idx] = static_cast<double>(pkt->re[k]);
        im[idx] = static_cast<double>(pkt->im[k]);

        // Completar con el bin conjugado simétrico
        // DC (idx=0) y Nyquist (idx=N/2) son reales: no tienen par
        if (idx > 0 && idx < N / 2) {
            re[N - idx] =  static_cast<double>(pkt->re[k]);
            im[N - idx] = -static_cast<double>(pkt->im[k]);
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Tarea Core 1 (alta prioridad): recepción y parseo de paquetes
// ────────────────────────────────────────────────────────────
void taskReceive(void* /*param*/) {
    for (;;) {
        // 1. Sincronizar con cabecera [0xAA][0x55]
        for (;;) {
            while (Serial2.available() < 1) taskYIELD();
            if (Serial2.read() != ESP_HDR_A) continue;
            while (Serial2.available() < 1) taskYIELD();
            if (Serial2.peek() == ESP_HDR_B) { Serial2.read(); break; }
        }

        // 2. CRC acumulado (cubre todo lo que sigue al header)
        uint8_t crc = 0;

        auto rb = [&](uint8_t& b) {
            while (!Serial2.available()) taskYIELD();
            b = Serial2.read();
            crc ^= b;
        };
        auto rbuf = [&](void* dst, size_t len) {
            uint8_t* p = reinterpret_cast<uint8_t*>(dst);
            for (size_t i = 0; i < len; i++) rb(p[i]);
        };

        // 3. Leer N del bloque
        uint16_t blockN = 0;
        rbuf(&blockN, sizeof(blockN));
        if (blockN != N) continue;  // descarta si no coincide

        RecvPacket* pkt = new RecvPacket();

        // 4. Muestras originales
        rbuf(pkt->original, N * sizeof(int16_t));

        // 5. K
        rbuf(&pkt->numCoeffs, sizeof(pkt->numCoeffs));
        if (pkt->numCoeffs > N / 2 + 1) { delete pkt; continue; }

        // 6. Coeficientes
        for (uint16_t k = 0; k < pkt->numCoeffs; k++) {
            rbuf(&pkt->indices[k], sizeof(uint16_t));
            rbuf(&pkt->re[k],      sizeof(float));
            rbuf(&pkt->im[k],      sizeof(float));
        }

        // 7. Validar CRC
        uint8_t rxCrc;
        while (!Serial2.available()) taskYIELD();
        rxCrc = Serial2.read();
        if (rxCrc != crc) { delete pkt; continue; }

        // 8. Enviar copia a taskAudio y puntero original a taskMetrics
        RecvPacket* pktAudio = new RecvPacket(*pkt);
        if (xQueueSend(audioQueue,   &pktAudio, pdMS_TO_TICKS(50)) != pdTRUE) {
            delete pktAudio;
        }
        if (xQueueSend(metricsQueue, &pkt,      pdMS_TO_TICKS(50)) != pdTRUE) {
            delete pkt;
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Tarea Core 0: IFFT + salida por DAC  (Receptor 1)
// ────────────────────────────────────────────────────────────
void taskAudio(void* /*param*/) {
    static double     re[N], im[N];
    static arduinoFFT FFT;
    RecvPacket* pkt;

    for (;;) {
        if (xQueueReceive(audioQueue, &pkt, portMAX_DELAY) != pdTRUE) continue;

        // Reconstruir espectro completo con simetría hermítica
        reconstructSpectrum(pkt, re, im);

        // IFFT
        FFT.Compute(re, im, N, FFT_REVERSE);

        // Normalizar y enviar muestra a muestra al DAC
        // int16 original: rango [-32768, 32767] → DAC 8-bit: rango [0, 255]
        for (int i = 0; i < N; i++) {
            double s = re[i] / N;
            int dacVal = static_cast<int>((s + 32768.0) * 255.0 / 65535.0);
            dacWrite(SPEAKER_PIN, constrain(dacVal, 0, 255));
            delayMicroseconds(1000000 / SAMPLE_RATE);  // 125 µs @ 8 kHz
        }

        delete pkt;
    }
}

// ────────────────────────────────────────────────────────────
//  Tarea Core 1 (baja prioridad): IFFT + métricas + LCD  (Receptor 2)
// ────────────────────────────────────────────────────────────
void taskMetrics(void* /*param*/) {
    static double     re[N], im[N];
    static arduinoFFT FFT;
    RecvPacket* pkt;

    for (;;) {
        if (xQueueReceive(metricsQueue, &pkt, portMAX_DELAY) != pdTRUE) continue;

        // Reconstruir espectro y aplicar IFFT
        reconstructSpectrum(pkt, re, im);
        FFT.Compute(re, im, N, FFT_REVERSE);

        // ── Calcular métricas ───────────────────────────────
        double mse         = 0.0;
        double energyOrig  = 0.0;
        double energyRecon = 0.0;
        double errorEnergy = 0.0;

        for (int i = 0; i < N; i++) {
            double orig  = static_cast<double>(pkt->original[i]);
            double recon = re[i] / N;
            double err   = orig - recon;

            mse         += err * err;
            energyOrig  += orig  * orig;
            energyRecon += recon * recon;
            errorEnergy += err   * err;
        }
        mse /= N;

        // Energía preservada (%)
        double energyPct = (energyOrig > 0.0)
                           ? (energyRecon / energyOrig) * 100.0
                           : 0.0;

        // SNR en dB (señal-a-error)
        double snr = (errorEnergy > 1e-9)
                     ? 10.0 * log10(energyOrig / errorEnergy)
                     : 99.9;

        // ── Debug por USB-Serial ────────────────────────────
        Serial.print("[RX] MSE="); Serial.print(mse, 2);
        Serial.print("  En=");     Serial.print(energyPct, 1); Serial.print("%");
        Serial.print("  SNR=");    Serial.print(snr, 1);       Serial.print("dB");
        Serial.print("  K=");      Serial.println(pkt->numCoeffs);

        // ── Mostrar en LCD 20×4 ────────────────────────────
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
        lcd.print(pkt->numCoeffs);
        lcd.print("/");
        lcd.print(N / 2 + 1);
        lcd.print(" bins");

        delete pkt;
    }
}

// ────────────────────────────────────────────────────────────
//  Setup
// ────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial2.begin(UART2_BAUD, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);

    // LCD
    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0);
    lcd.print("ESP32 Receptor");
    lcd.setCursor(0, 1);
    lcd.print("Esperando datos...");

    // Silencio inicial en el DAC
    dacWrite(SPEAKER_PIN, 128);

    // Colas de punteros (hasta 4 paquetes pendientes por cola)
    audioQueue   = xQueueCreate(4, sizeof(RecvPacket*));
    metricsQueue = xQueueCreate(4, sizeof(RecvPacket*));

    // taskReceive → Core 1, prioridad alta (no perder bytes UART)
    xTaskCreatePinnedToCore(taskReceive, "RX",
                            8192, NULL, 3, NULL, 1);

    // taskAudio → Core 0 exclusivo (delayMicroseconds no bloquea otro core)
    xTaskCreatePinnedToCore(taskAudio, "Audio",
                            8192, NULL, 2, NULL, 0);

    // taskMetrics → Core 1, prioridad baja (corre entre paquetes)
    xTaskCreatePinnedToCore(taskMetrics, "Metrics",
                            8192, NULL, 1, NULL, 1);
}

void loop() {
    vTaskDelay(portMAX_DELAY);
}
