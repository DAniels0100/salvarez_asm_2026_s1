// ============================================================
//  ifft_audio.ino  —  ESP32 #2: Receptor (PC como puente)
// ============================================================
//  Recibe coeficientes comprimidos directamente del PC por
//  USB-Serial (el PC los lee del ESP32 #1 y los reenvia aqui).
//
//  Arquitectura FreeRTOS:
//    Core 1 — taskReceive : parsea paquetes del PC (USB)
//    Core 0 — taskAudio   : IFFT -> DAC -> Speaker
//    Core 1 — taskMetrics : IFFT -> MSE/Energia/SNR -> LCD
//
//  Protocolo PC -> ESP32 (USB, 115200 baud):
//    [0xAA][0x55]
//    N            : uint16_t
//    original[N]  : int16_t x N
//    K            : uint16_t
//    K x (index:u16, re:f32, im:f32)
//    crc          : uint8_t
//
//  Conexion hardware:
//    GPIO 25 (DAC1) --> filtro RC --> TPA2005D1 --> Speaker
//    GPIO 21 (SDA)  --> LCD I2C SDA
//    GPIO 22 (SCL)  --> LCD I2C SCL
//
//  Libreria requerida: arduinoFFT v2.x
// ============================================================

#include <Arduino.h>
#include <arduinoFFT.h>
#include <LiquidCrystal_I2C.h>

// -- Parametros ---------------------------------------------
#define N            256
#define SAMPLE_RATE  8000.0

// -- Pines --------------------------------------------------
#define SPEAKER_PIN  25

// -- Cabeceras de protocolo ---------------------------------
#define RX_HDR_A  0xAA
#define RX_HDR_B  0x55

// -- Paquete recibido --------------------------------------
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
//  Helper: reconstruir espectro con simetria hermitica
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
//  Tarea Core 1: recepcion y parseo desde USB-Serial
// ----------------------------------------------------------
void taskReceive(void* /*param*/) {
    for (;;) {
        // 1. Sincronizar con cabecera [0xAA][0x55]
        for (;;) {
            while (Serial.available() < 1) taskYIELD();
            if (Serial.read() != RX_HDR_A) continue;
            while (Serial.available() < 1) taskYIELD();
            if (Serial.peek() == RX_HDR_B) { Serial.read(); break; }
        }

        uint8_t crc = 0;

        auto rb = [&](uint8_t& b) {
            while (!Serial.available()) taskYIELD();
            b = Serial.read();
            crc ^= b;
        };
        auto rbuf = [&](void* dst, size_t len) {
            uint8_t* p = reinterpret_cast<uint8_t*>(dst);
            for (size_t i = 0; i < len; i++) rb(p[i]);
        };

        uint16_t blockN = 0;
        rbuf(&blockN, sizeof(blockN));
        if (blockN != N) continue;

        RecvPacket* pkt = new RecvPacket();
        rbuf(pkt->original, N * sizeof(int16_t));
        rbuf(&pkt->numCoeffs, sizeof(pkt->numCoeffs));
        if (pkt->numCoeffs > N / 2 + 1) {
            delete pkt; continue;
        }

        for (uint16_t k = 0; k < pkt->numCoeffs; k++) {
            rbuf(&pkt->indices[k], sizeof(uint16_t));
            rbuf(&pkt->re[k],      sizeof(float));
            rbuf(&pkt->im[k],      sizeof(float));
        }

        uint8_t rxCrc;
        while (!Serial.available()) taskYIELD();
        rxCrc = Serial.read();
        if (rxCrc != crc) {
            delete pkt; continue;
        }

        // Distribuir copia a ambas tareas
        RecvPacket* pktAudio = new RecvPacket(*pkt);
        if (xQueueSend(audioQueue, &pktAudio, pdMS_TO_TICKS(50)) != pdTRUE) {
            delete pktAudio;
        }
        if (xQueueSend(metricsQueue, &pkt, pdMS_TO_TICKS(50)) != pdTRUE) {
            delete pkt;
        }
    }
}

// ----------------------------------------------------------
//  Tarea Core 0: IFFT + DAC -> Speaker (Receptor 1)
// ----------------------------------------------------------
void taskAudio(void* /*param*/) {
    static double re[N], im[N];
    static ArduinoFFT<double> FFT(re, im, N, SAMPLE_RATE);
    RecvPacket* pkt;

    for (;;) {
        if (xQueueReceive(audioQueue, &pkt, portMAX_DELAY) != pdTRUE) continue;

        reconstructSpectrum(pkt, re, im);
        FFT.compute(FFTDirection::Reverse);

        // Mapeo con ganancia — la senal reconstruida suele estar
        // en un rango pequeno (cientos a pocos miles), no en ±32768.
        // Escalamos dinamicamente buscando el maximo del bloque.
        double maxAbs = 1.0;
        for (int i = 0; i < N; i++) {
            double v = fabs(re[i]);
            if (v > maxAbs) maxAbs = v;
        }
        // Factor que lleva maxAbs a 127 (rango util del DAC)
        double gain = 127.0 / maxAbs;

        for (int i = 0; i < N; i++) {
            double s = re[i] * gain;                  // [-127, 127]
            int dacVal = (int)(s + 128.0);            // [0, 255]
            dacWrite(SPEAKER_PIN, (uint8_t)constrain(dacVal, 0, 255));
            delayMicroseconds(125);                   // 8 kHz

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

        // DIAGNOSTICO: imprimir algunos valores para ver la escala
        static int diagCount = 0;
        if (diagCount++ < 3) {
            Serial.print("[DIAG] original[0..3]: ");
            Serial.print(pkt->original[0]); Serial.print(",");
            Serial.print(pkt->original[1]); Serial.print(",");
            Serial.print(pkt->original[2]); Serial.print(",");
            Serial.println(pkt->original[3]);

            Serial.print("[DIAG] re[0..3] crudo: ");
            Serial.print(re[0], 2); Serial.print(",");
            Serial.print(re[1], 2); Serial.print(",");
            Serial.print(re[2], 2); Serial.print(",");
            Serial.println(re[3], 2);

            // Ya no dividimos por N — los valores crudos son correctos
        }

        double mse = 0.0, energyOrig = 0.0, energyRecon = 0.0, errorEnergy = 0.0;

        for (int i = 0; i < N; i++) {
            double orig  = static_cast<double>(pkt->original[i]);
            double recon = re[i];   // IFFT v2.x ya normaliza
            double err   = orig - recon;

            mse         += err  * err;
            energyOrig  += orig * orig;
            energyRecon += recon * recon;
            errorEnergy += err  * err;
        }
        mse /= N;

        double energyPct = (energyOrig > 0.0)
                           ? (energyRecon / energyOrig) * 100.0 : 0.0;
        double snr = (errorEnergy > 1e-9)
                     ? 10.0 * log10(energyOrig / errorEnergy) : 99.9;

        // LCD
        lcd.clear();
        lcd.setCursor(0, 0); lcd.print("MSE: ");     lcd.print(mse, 2);
        lcd.setCursor(0, 1); lcd.print("Energia: "); lcd.print(energyPct, 1); lcd.print("%");
        lcd.setCursor(0, 2); lcd.print("SNR: ");     lcd.print(snr, 1); lcd.print(" dB");
        lcd.setCursor(0, 3); lcd.print("K=");        lcd.print(pkt->numCoeffs);
                              lcd.print("/"); lcd.print(N / 2 + 1); lcd.print(" bins");

        delete pkt;
    }
}

// ----------------------------------------------------------
//  Setup
// ----------------------------------------------------------
void setup() {
    Serial.begin(115200);

    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0);
    lcd.print("ESP32 Receptor");
    lcd.setCursor(0, 1);
    lcd.print("Esperando datos...");

    dacWrite(SPEAKER_PIN, 128);  // silencio inicial

    audioQueue   = xQueueCreate(4, sizeof(RecvPacket*));
    metricsQueue = xQueueCreate(4, sizeof(RecvPacket*));

    xTaskCreatePinnedToCore(taskReceive, "RX",     8192, NULL, 3, NULL, 1);
    xTaskCreatePinnedToCore(taskAudio,   "Audio",  8192, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(taskMetrics, "Metrics", 8192, NULL, 1, NULL, 1);
}

void loop() {
    vTaskDelay(portMAX_DELAY);
}
