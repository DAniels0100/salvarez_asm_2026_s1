// ============================================================
//  ifft_audio.ino  —  ESP32 #2: Receptor (PC como puente)
// ============================================================
//  Recibe coeficientes comprimidos directamente del PC por
//  USB-Serial (el PC los lee del ESP32 #1 y los reenvia aqui).
//
//  Arquitectura FreeRTOS:
//    Core 0 — taskReceive : parsea paquetes del PC (USB)
//    Core 1 — taskAudio   : desempaqueta (IFFT) y reproduce al final
//    Core 0 — taskMetrics : MSE/Energia/SNR global -> LCD
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
// Debe coincidir con el TX / send_audio.py
#define MAX_BLOCKS   187

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
    uint16_t totalBlocks;
    double   mse;
    double   energyPct;
    double   snr;
};
static QueueHandle_t metricsQueue;

// Estado de una "sesion" de transmision
static volatile bool     g_sessionActive = false;
static volatile bool     g_rxEnded = false;
static volatile uint16_t g_expectedBlocks = 0;
static volatile uint16_t g_receivedBlocks = 0;
static volatile uint16_t g_decompressedBlocks = 0;
static int16_t* g_origAll = nullptr;   // [expectedBlocks * N]
static int16_t* g_reconAll = nullptr;  // [expectedBlocks * N]

static SemaphoreHandle_t semSessionStart;
static SemaphoreHandle_t semRxEnd;

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

    auto resetSession = [&]() {
        g_sessionActive = false;
        g_rxEnded = false;
        g_expectedBlocks = 0;
        g_receivedBlocks = 0;
        g_decompressedBlocks = 0;
        if (g_origAll) { free(g_origAll); g_origAll = nullptr; }
        if (g_reconAll) { free(g_reconAll); g_reconAll = nullptr; }
        // Vaciar FIFO (si hubiese basura)
        for (;;) {
            RecvPacket* p = popPacket(0);
            if (!p) break;
            delete p;
        }
        // Re-inicializar semáforos de la FIFO
        if (semItems) {
            while (xSemaphoreTake(semItems, 0) == pdTRUE) {}
        }
        if (semSpaces) {
            while (xSemaphoreTake(semSpaces, 0) == pdTRUE) {}
            for (int i = 0; i < PACKET_BUFFER_LEN; i++) xSemaphoreGive(semSpaces);
        }
        taskENTER_CRITICAL(&packetMux);
        packetHead = packetTail = packetCount = 0;
        taskEXIT_CRITICAL(&packetMux);
    };

    resetSession();
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

        // ---- Control frames ----
        // Formato:
        //  [AA55][blockN=0x0000][cmd:u8][(totalBlocks:u16 si cmd=1)][crc:u8]
        if (blockN == 0) {
            uint8_t cmd = 0;
            rb(cmd);
            uint16_t totalBlocks = 0;
            if (cmd == 1) {
                rbuf(&totalBlocks, sizeof(totalBlocks));
            }
            uint8_t rxCrc;
            while (!Serial.available()) taskYIELD();
            rxCrc = Serial.read();
            if (rxCrc != crc) {
                continue;
            }

            if (cmd == 1) {
                // START
                resetSession();
                if (totalBlocks == 0) totalBlocks = 1;
                if (totalBlocks > MAX_BLOCKS) totalBlocks = MAX_BLOCKS;
                g_expectedBlocks = totalBlocks;

                g_origAll = (int16_t*)malloc((size_t)g_expectedBlocks * N * sizeof(int16_t));
                g_reconAll = (int16_t*)malloc((size_t)g_expectedBlocks * N * sizeof(int16_t));
                if (!g_origAll || !g_reconAll) {
                    if (g_origAll) { free(g_origAll); g_origAll = nullptr; }
                    if (g_reconAll) { free(g_reconAll); g_reconAll = nullptr; }
                    // No hay RAM -> descartar sesion
                    continue;
                }

                g_sessionActive = true;
                g_rxEnded = false;
                xSemaphoreGive(semSessionStart);

                // Primer update al LCD: progreso 0/N
                MetricsFrame mf;
                mf.seq = 0;
                mf.totalBlocks = g_expectedBlocks;
                mf.mse = 0.0;
                mf.energyPct = 0.0;
                mf.snr = 0.0;
                (void)xQueueSend(metricsQueue, &mf, 0);
            } else if (cmd == 2) {
                // END
                if (g_sessionActive) {
                    g_rxEnded = true;
                    xSemaphoreGive(semRxEnd);
                }
            }
            continue;
        }

        // ---- Data frames ----
        if (blockN != N) continue;
        if (!g_sessionActive || !g_origAll || !g_reconAll) {
            // Si llegan datos sin START, los descartamos
            continue;
        }

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

        // Guardar originales en el buffer global (por orden de llegada)
        uint16_t blk = g_receivedBlocks;
        if (blk < g_expectedBlocks) {
            memcpy(&g_origAll[(size_t)blk * N], pkt->original, N * sizeof(int16_t));
            pkt->seq = blk; // re-etiquetar con indice de bloque
            g_receivedBlocks++;
        } else {
            // Llegaron mas bloques de los esperados
            delete pkt;
            continue;
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

    // Acumuladores de métricas globales (toda la sesion)
    double sumErr2 = 0.0;
    double sumOrig2 = 0.0;
    double sumRecon2 = 0.0;
    double maxAbsRecon = 1.0;
    uint32_t sampleCount = 0;

    for (;;) {
        // Esperar inicio de sesion
        if (xSemaphoreTake(semSessionStart, portMAX_DELAY) != pdTRUE) continue;

        // Reset acumuladores
        sumErr2 = 0.0;
        sumOrig2 = 0.0;
        sumRecon2 = 0.0;
        maxAbsRecon = 1.0;
        sampleCount = 0;

        // --- Fase A: desempaquetar (IFFT) mientras llegan datos ---
        for (;;) {
            // Tomar paquetes en orden; si se termina RX, salimos cuando se hayan procesado todos
            pkt = popPacket(pdMS_TO_TICKS(50));
            if (!pkt) {
                // Si ya recibimos END y no hay mas en FIFO, terminamos el desempaquetado
                if (g_rxEnded) break;
                // Aun no hay paquete disponible
                continue;
            }

            reconstructSpectrum(pkt, re, im);
            FFT.compute(FFTDirection::Reverse);

            const uint16_t blk = (uint16_t)pkt->seq;
            if (g_reconAll && blk < g_expectedBlocks) {
                for (int i = 0; i < N; i++) {
                    double recon = re[i];
                    double orig = (double)g_origAll[(size_t)blk * N + i];
                    double err = orig - recon;

                    sumErr2 += err * err;
                    sumOrig2 += orig * orig;
                    sumRecon2 += recon * recon;
                    sampleCount++;

                    double a = fabs(recon);
                    if (a > maxAbsRecon) maxAbsRecon = a;

                    // Guardar reconstruido (clamp a int16)
                    double cl = recon;
                    if (cl > 32767.0) cl = 32767.0;
                    if (cl < -32768.0) cl = -32768.0;
                    g_reconAll[(size_t)blk * N + i] = (int16_t)lrint(cl);
                }

                g_decompressedBlocks++;

                // Actualización de métricas cada ~10 bloques (métrica global parcial)
                if ((g_decompressedBlocks % 10) == 0 && sampleCount > 0) {
                    MetricsFrame mf;
                    mf.seq = g_decompressedBlocks;
                    mf.totalBlocks = g_expectedBlocks;
                    mf.mse = sumErr2 / (double)sampleCount;
                    mf.energyPct = (sumOrig2 > 0.0) ? (sumRecon2 / sumOrig2) * 100.0 : 0.0;
                    mf.snr = (sumErr2 > 1e-9) ? 10.0 * log10(sumOrig2 / sumErr2) : 99.9;
                    (void)xQueueSend(metricsQueue, &mf, 0);
                }
            }

            delete pkt;
        }

        // Esperar a que llegue END (si aun no llegó)
        if (!g_rxEnded) {
            (void)xSemaphoreTake(semRxEnd, portMAX_DELAY);
        }

        // Métricas finales (globales)
        if (sampleCount > 0) {
            MetricsFrame mf;
            mf.seq = g_expectedBlocks;
            mf.totalBlocks = g_expectedBlocks;
            mf.mse = sumErr2 / (double)sampleCount;
            mf.energyPct = (sumOrig2 > 0.0) ? (sumRecon2 / sumOrig2) * 100.0 : 0.0;
            mf.snr = (sumErr2 > 1e-9) ? 10.0 * log10(sumOrig2 / sumErr2) : 99.9;
            (void)xQueueSend(metricsQueue, &mf, portMAX_DELAY);
        }

        // --- Fase B: reproducir SOLO cuando terminó la transmisión ---
        // Ganancia global para DAC: maxAbsRecon -> 127
        double gain = 127.0 / maxAbsRecon;
        // Reproducimos solo lo que realmente se alcanzó a desempaquetar.
        // Si hubo drops, esto evita leer basura/memoria sin escribir.
        const uint16_t blocksToPlay = g_decompressedBlocks;
        for (uint16_t blk = 0; blk < blocksToPlay; blk++) {
            for (int i = 0; i < N; i++) {
                double recon = (double)g_reconAll[(size_t)blk * N + i];
                double s = recon * gain;
                int dacVal = (int)(s + 128.0);
                dacWrite(SPEAKER_PIN, (uint8_t)constrain(dacVal, 0, 255));
                delayMicroseconds(125);
            }
            taskYIELD();
        }

        // Termina la sesion: liberar buffers
        if (g_origAll) { free(g_origAll); g_origAll = nullptr; }
        if (g_reconAll) { free(g_reconAll); g_reconAll = nullptr; }
        g_sessionActive = false;
        g_rxEnded = false;
        g_expectedBlocks = 0;
        g_receivedBlocks = 0;
        g_decompressedBlocks = 0;
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
        lcd.setCursor(0, 3);
        lcd.print("Blk: ");
        lcd.print((unsigned long)mf.seq);
        lcd.print("/");
        lcd.print(mf.totalBlocks);
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

    semSessionStart = xSemaphoreCreateBinary();
    semRxEnd = xSemaphoreCreateBinary();

    // Core 0: RX + LCD (UI)
    // Core 1: Audio (desempaquetado + reproduccion)
    xTaskCreatePinnedToCore(taskReceive, "RX",      8192, NULL, 3, NULL, 0);
    xTaskCreatePinnedToCore(taskMetrics, "Metrics", 8192, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(taskAudio,   "Audio",   8192, NULL, 2, NULL, 1);
}

void loop() {
    vTaskDelay(portMAX_DELAY);
}
