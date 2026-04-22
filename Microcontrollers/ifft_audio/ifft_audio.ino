// ============================================================
//  esp32_ifft_audio.ino  —  ESP32 #2: Receptor Combinado
// ============================================================
//  Implementa Receptor 1 (audio) y Receptor 2 (metricas) en un
//  solo ESP32 usando FreeRTOS dual-core.
//
//  Arquitectura:
//    Core 1 — taskReceive : parsea paquetes UART2 (alta prioridad)
//    Core 0 — taskAudio   : IFFT -> DAC -> Speaker  (Receptor 1)
//    Core 1 — taskMetrics : IFFT -> MSE/Energia/SNR -> LCD (Receptor 2)
//
//  Protocolo recibido desde ESP32 #1 (UART2, 921600 baud):
//    [0xAA][0x55] N:u16 original:Nxi16 K:u16 Kx(idx:u16,re:f32,im:f32) crc:u8
//
//  Conexion hardware:
//    GPIO 16 (RX2) <-- GPIO 17 (TX2) de ESP32 #1
//    GND           <-- GND           de ESP32 #1
//    GPIO 25 (DAC1) --> RC LPF --> TPA2005D1 --> Speaker
//    GPIO 21 (SDA) --> LCD I2C SDA
//    GPIO 22 (SCL) --> LCD I2C SCL
//
//  Libreria requerida: arduinoFFT v2.x
// ============================================================

#include <Arduino.h>
#include <arduinoFFT.h>
#include <LiquidCrystal_I2C.h>

// -- Parametros del sistema ---------------------------------
#define N            256
#define SAMPLE_RATE  8000.0

// -- Pines --------------------------------------------------
#define SPEAKER_PIN  25
#define UART2_RX_PIN 16
#define UART2_TX_PIN 17
#define UART2_BAUD 57600

// -- Cabeceras de protocolo ---------------------------------
#define ESP_HDR_A  0xAA
#define ESP_HDR_B  0x55

// -- Paquete recibido (heap) -------------------------------
struct RecvPacket {
    int16_t  original[N];
    uint16_t numCoeffs;
    uint16_t indices[N / 2 + 1];
    float    re[N / 2 + 1];
    float    im[N / 2 + 1];
};

// -- Colas FreeRTOS ----------------------------------------
static QueueHandle_t audioQueue;
static QueueHandle_t metricsQueue;

// -- LCD 20x4 I2C ------------------------------------------
static LiquidCrystal_I2C lcd(0x27, 20, 4);

// ----------------------------------------------------------
//  Helper: reconstruir espectro completo con simetria hermitica
//  Para senal real: X[N-k] = conj(X[k])
// ----------------------------------------------------------
static void reconstructSpectrum(const RecvPacket* pkt,
                                 double* re, double* im) {
    memset(re, 0, N * sizeof(double));
    memset(im, 0, N * sizeof(double));

    for (uint16_t k = 0; k < pkt->numCoeffs; k++) {
        uint16_t idx = pkt->indices[k];
        if (idx > N / 2) continue;

        re[idx] =  static_cast<double>(pkt->re[k]);
        im[idx] =  static_cast<double>(pkt->im[k]);

        if (idx > 0 && idx < N / 2) {
            re[N - idx] =  static_cast<double>(pkt->re[k]);
            im[N - idx] = -static_cast<double>(pkt->im[k]);
        }
    }
}

// ----------------------------------------------------------
//  Tarea Core 1: recepcion y parseo UART2
// ----------------------------------------------------------
// ----------------------------------------------------------
//  Receptor ROBUSTO con buffer + resync
// ----------------------------------------------------------
void taskReceive(void* /*param*/) {

    static uint8_t rxBuffer[4096];
    static int bufferLen = 0;

    for (;;) {

        // 🔹 1. Leer todo lo disponible sin bloquear
        while (Serial2.available()) {
            if (bufferLen < sizeof(rxBuffer)) {
                rxBuffer[bufferLen++] = Serial2.read();
            } else {
                // overflow -> reset
                bufferLen = 0;
            }
        }

        // 🔹 2. Procesar buffer
        int i = 0;

        while (i <= bufferLen - 4) {

            // Buscar header
            if (rxBuffer[i] != ESP_HDR_A || rxBuffer[i+1] != ESP_HDR_B) {
                i++;
                continue;
            }

            // Leer payloadSize
            uint16_t payloadSize = rxBuffer[i+2] | (rxBuffer[i+3] << 8);

            // Validación básica
            if (payloadSize < 10 || payloadSize > 2000) {
                i++; // avanzar 1 byte (resync)
                continue;
            }

            // Verificar si ya llegó todo el paquete
            int totalSize = 2 + 2 + payloadSize + 1; // hdr + size + payload + crc

            if (i + totalSize > bufferLen) {
                break; // esperar más datos
            }

            // Calcular CRC
            uint8_t crcCalc = 0;
            for (int j = 0; j < payloadSize; j++) {
                crcCalc ^= rxBuffer[i + 4 + j];
            }

            uint8_t crcRx = rxBuffer[i + 4 + payloadSize];

            if (crcCalc != crcRx) {
                Serial.println("[RX] CRC error -> resync");
                i++; // avanzar 1 byte y reintentar
                continue;
            }

            // 🔥 Paquete válido
            uint8_t* payload = &rxBuffer[i + 4];

            int idx = 0;

            auto rbuf = [&](void* dst, size_t len) {
                memcpy(dst, &payload[idx], len);
                idx += len;
            };

            uint16_t blockN;
            rbuf(&blockN, sizeof(blockN));

            if (blockN != N) {
                Serial.println("[RX] N invalido");
                i += totalSize;
                continue;
            }

            RecvPacket* pkt = new RecvPacket();

            rbuf(pkt->original, N * sizeof(int16_t));
            rbuf(&pkt->numCoeffs, sizeof(uint16_t));

            if (pkt->numCoeffs > N/2+1) {
                Serial.println("[RX] K invalido");
                delete pkt;
                i += totalSize;
                continue;
            }

            for (uint16_t k = 0; k < pkt->numCoeffs; k++) {
                rbuf(&pkt->indices[k], sizeof(uint16_t));
                rbuf(&pkt->re[k], sizeof(float));
                rbuf(&pkt->im[k], sizeof(float));
            }

            if (idx != payloadSize) {
                Serial.println("[RX] Parse mismatch");
                delete pkt;
                i += totalSize;
                continue;
            }

            // Debug OK
            Serial.print("[RX] OK packet K=");
            Serial.println(pkt->numCoeffs);

            // Enviar a colas
            RecvPacket* pktAudio = new RecvPacket(*pkt);

            if (xQueueSend(audioQueue, &pktAudio, 0) != pdTRUE) {
                delete pktAudio;
            }

            if (xQueueSend(metricsQueue, &pkt, 0) != pdTRUE) {
                delete pkt;
            }

            // avanzar al siguiente paquete
            i += totalSize;
        }

        // 🔹 3. Compactar buffer (eliminar lo ya procesado)
        if (i > 0) {
            memmove(rxBuffer, rxBuffer + i, bufferLen - i);
            bufferLen -= i;
        }

        vTaskDelay(1);
    }
}

// ----------------------------------------------------------
//  Tarea Core 0: IFFT + DAC -> Speaker (Receptor 1)
// ----------------------------------------------------------
void taskAudio(void* /*param*/) {
    static double re[N], im[N];
    // v2.x: instancia con punteros a buffers
    static ArduinoFFT<double> FFT(re, im, N, SAMPLE_RATE);
    RecvPacket* pkt;

    for (;;) {
        if (xQueueReceive(audioQueue, &pkt, portMAX_DELAY) != pdTRUE) continue;

        reconstructSpectrum(pkt, re, im);

        // IFFT v2.x
        FFT.compute(FFTDirection::Reverse);

        // Reproducir muestras al DAC
        for (int i = 0; i < N; i++) {
            // IFFT en arduinoFFT v2.x: dividir por N para normalizar
            double s = re[i] / static_cast<double>(N);
            // Mapeo int16 [-32768, 32767] -> DAC 8-bit [0, 255]
            int dacVal = static_cast<int>((s + 32768.0) * 255.0 / 65535.0);
            dacWrite(SPEAKER_PIN, (uint8_t)constrain(dacVal, 0, 255));
            delayMicroseconds(80);  // 8 kHz

            if ((i & 0x1F) == 0x1F) taskYIELD();
        }

        delete pkt;
    }
}

// ----------------------------------------------------------
//  Tarea Core 1: IFFT + metricas + LCD (Receptor 2)
// ----------------------------------------------------------
void taskMetrics(void* /*param*/) {
    static double re[N], im[N];
    static ArduinoFFT<double> FFT(re, im, N, SAMPLE_RATE);
    RecvPacket* pkt;

    for (;;) {
        if (xQueueReceive(metricsQueue, &pkt, portMAX_DELAY) != pdTRUE) continue;

        reconstructSpectrum(pkt, re, im);
        FFT.compute(FFTDirection::Reverse);

        // Metricas
        double mse         = 0.0;
        double energyOrig  = 0.0;
        double energyRecon = 0.0;
        double errorEnergy = 0.0;

        for (int i = 0; i < N; i++) {
            double orig  = static_cast<double>(pkt->original[i]);
            double recon = re[i] / static_cast<double>(N);
            double err   = orig - recon;

            mse         += err  * err;
            energyOrig  += orig * orig;
            energyRecon += recon * recon;
            errorEnergy += err  * err;
        }
        mse /= N;

        double energyPct = (energyOrig > 0.0)
                           ? (energyRecon / energyOrig) * 100.0
                           : 0.0;

        double snr = (errorEnergy > 1e-9)
                     ? 10.0 * log10(energyOrig / errorEnergy)
                     : 99.9;

        // Debug serial
        Serial.print("[RX] MSE="); Serial.print(mse, 2);
        Serial.print("  En=");     Serial.print(energyPct, 1); Serial.print("%");
        Serial.print("  SNR=");    Serial.print(snr, 1);       Serial.print("dB");
        Serial.print("  K=");      Serial.println(pkt->numCoeffs);

        // LCD 20x4
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


// ----------------------------------------------------------
//  Setup
// ----------------------------------------------------------
void setup() {
    Serial.begin(115200);
    Serial2.begin(UART2_BAUD, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);

    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0);
    lcd.print("ESP32 Receptor");
    lcd.setCursor(0, 1);
    lcd.print("Esperando datos...");

    Serial.println("=== ESP32 Receptor IFFT + Metricas ===");
    Serial.println("Esperando paquetes del Transmisor por UART2...");

    dacWrite(SPEAKER_PIN, 128);  // silencio inicial

    audioQueue   = xQueueCreate(4, sizeof(RecvPacket*));
    metricsQueue = xQueueCreate(4, sizeof(RecvPacket*));

    xTaskCreatePinnedToCore(taskReceive, "RX",
                            8192, NULL, 3, NULL, 1);

    xTaskCreatePinnedToCore(taskAudio, "Audio",
                            8192, NULL, 2, NULL, 0);

    xTaskCreatePinnedToCore(taskMetrics, "Metrics",
                            8192, NULL, 1, NULL, 1);
}

void loop() {
    vTaskDelay(portMAX_DELAY);
}
