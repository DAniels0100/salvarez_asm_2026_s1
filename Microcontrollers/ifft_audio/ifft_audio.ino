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
    uint32_t seq;
    int16_t  original[N];
    uint16_t numCoeffs;
    uint16_t indices[N / 2 + 1];
    float    re[N / 2 + 1];
    float    im[N / 2 + 1];
};

// -- Colas FreeRTOS ----------------------------------------
// Guardamos paquetes recibidos en un array FIFO (ring buffer)
// para asegurar descompresion/IFFT en el mismo orden de llegada.
#define PACKET_BUFFER_LEN  32
static RecvPacket* packetBuffer[PACKET_BUFFER_LEN];
static volatile uint16_t packetHead = 0;
static volatile uint16_t packetTail = 0;
static volatile uint16_t packetCount = 0;
static portMUX_TYPE packetMux = portMUX_INITIALIZER_UNLOCKED;
static SemaphoreHandle_t semItems;
static SemaphoreHandle_t semSpaces;

// Cola liviana solo para métricas (evita duplicar paquetes grandes)
struct MetricsFrame {
    uint32_t seq;
    double   mse;
    double   energyPct;
    double   snr;
    uint16_t numCoeffs;
};
static QueueHandle_t metricsQueue;

static inline bool pushPacket(RecvPacket* pkt, TickType_t timeoutTicks) {
    if (xSemaphoreTake(semSpaces, timeoutTicks) != pdTRUE) return false;
    taskENTER_CRITICAL(&packetMux);
    packetBuffer[packetHead] = pkt;
    packetHead = (uint16_t)((packetHead + 1) % PACKET_BUFFER_LEN);
    packetCount++;
    taskEXIT_CRITICAL(&packetMux);
    xSemaphoreGive(semItems);
    return true;
}

static inline RecvPacket* popPacket(TickType_t timeoutTicks) {
    if (xSemaphoreTake(semItems, timeoutTicks) != pdTRUE) return nullptr;
    taskENTER_CRITICAL(&packetMux);
    RecvPacket* pkt = packetBuffer[packetTail];
    packetTail = (uint16_t)((packetTail + 1) % PACKET_BUFFER_LEN);
    packetCount--;
    taskEXIT_CRITICAL(&packetMux);
    xSemaphoreGive(semSpaces);
    return pkt;
}

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
    static uint32_t seqCounter = 0;
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
        pkt->seq = seqCounter++;
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

        // Guardar en FIFO (array) para procesar estrictamente en orden
        if (!pushPacket(pkt, pdMS_TO_TICKS(50))) {
            // buffer lleno -> descartamos el paquete más nuevo
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
        pkt = popPacket(portMAX_DELAY);
        if (!pkt) continue;

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

        // Calcular métricas y mandarlas a la tarea LCD (sin bloquear audio con I2C)
        double mse = 0.0, energyOrig = 0.0, energyRecon = 0.0, errorEnergy = 0.0;
        for (int i = 0; i < N; i++) {
            double orig  = static_cast<double>(pkt->original[i]);
            double recon = re[i];   // arduinoFFT v2.x: IFFT ya normaliza
            double err   = orig - recon;
            mse         += err  * err;
            energyOrig  += orig * orig;
            energyRecon += recon * recon;
            errorEnergy += err  * err;
        }
        mse /= N;
        double energyPct = (energyOrig > 0.0) ? (energyRecon / energyOrig) * 100.0 : 0.0;
        double snr = (errorEnergy > 1e-9) ? 10.0 * log10(energyOrig / errorEnergy) : 99.9;

        MetricsFrame mf;
        mf.seq = pkt->seq;
        mf.mse = mse;
        mf.energyPct = energyPct;
        mf.snr = snr;
        mf.numCoeffs = pkt->numCoeffs;
        (void)xQueueSend(metricsQueue, &mf, 0);

        delete pkt;
    }
}

// ----------------------------------------------------------
//  Tarea Core 1: IFFT + metricas + LCD (Receptor 2)
// ----------------------------------------------------------
void taskMetrics(void* /*param*/) {
    MetricsFrame mf;

    for (;;) {
        if (xQueueReceive(metricsQueue, &mf, portMAX_DELAY) != pdTRUE) continue;

        // LCD
        lcd.clear();
        lcd.setCursor(0, 0); lcd.print("MSE: ");     lcd.print(mf.mse, 2);
        lcd.setCursor(0, 1); lcd.print("Energia: "); lcd.print(mf.energyPct, 1); lcd.print("%");
        lcd.setCursor(0, 2); lcd.print("SNR: ");     lcd.print(mf.snr, 1); lcd.print(" dB");
        lcd.setCursor(0, 3); lcd.print("K=");        lcd.print(mf.numCoeffs);
                              lcd.print("/"); lcd.print(N / 2 + 1); lcd.print(" bins");
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

    // FIFO (array) de paquetes
    semItems  = xSemaphoreCreateCounting(PACKET_BUFFER_LEN, 0);
    semSpaces = xSemaphoreCreateCounting(PACKET_BUFFER_LEN, PACKET_BUFFER_LEN);
    metricsQueue = xQueueCreate(8, sizeof(MetricsFrame));

    xTaskCreatePinnedToCore(taskReceive, "RX",     8192, NULL, 3, NULL, 1);
    xTaskCreatePinnedToCore(taskAudio,   "Audio",  8192, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(taskMetrics, "Metrics", 8192, NULL, 1, NULL, 1);
}

void loop() {
    vTaskDelay(portMAX_DELAY);
}
