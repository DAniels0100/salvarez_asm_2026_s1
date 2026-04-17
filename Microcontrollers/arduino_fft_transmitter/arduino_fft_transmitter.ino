// ============================================================
//  esp32_fft_transmitter.ino  —  ESP32 #1: Transmisor
// ============================================================
//  Recibe bloques de muestras de audio desde el PC por USB-
//  Serial (script Python lee el .wav y los envía), calcula
//  la FFT, aplica compresión espectral adaptativa conservando
//  el 95 % de la energía y transmite únicamente los
//  coeficientes seleccionados a ESP32 #2 via UART2.
//
//  Arquitectura FreeRTOS (dual-core):
//    Core 0 — taskFFTCompress : recibe muestras, FFT, comprime
//    Core 1 — taskTransmit    : envía paquete binario por UART2
//
//  Protocolo PC → ESP32 (por USB-Serial, 115200 baud):
//    [0xFF][0xFE] + N × int16_t (little-endian)
//
//  Protocolo ESP32 → ESP32 (UART2, 921600 baud):
//    [0xAA][0x55]                  2 bytes  cabecera (no en CRC)
//    N            : uint16_t       2 bytes
//    original[N]  : int16_t×N      N×2 bytes
//    K            : uint16_t       2 bytes
//    Por cada coeficiente k=0..K-1:
//      index      : uint16_t       2 bytes  (0 … N/2)
//      re         : float          4 bytes
//      im         : float          4 bytes
//    crc          : uint8_t        1 byte   XOR de todo lo anterior al header
//
//  Conexión hardware:
//    ESP32 #1  GPIO 17 (TX2) ──► GPIO 16 (RX2) ESP32 #2
//    ESP32 #1  GND           ──► GND            ESP32 #2
// ============================================================

#include <Arduino.h>
#include <arduinoFFT.h>
#include <algorithm>

// ── Parámetros del sistema ──────────────────────────────────
#define N                  256     // Tamaño de bloque (potencia de 2)
#define SAMPLE_RATE        8000    // Hz — debe coincidir con el script Python
#define ENERGY_THRESHOLD   0.95f  // Fracción mínima de energía a preservar

// ── UART2 (inter-ESP32) ─────────────────────────────────────
#define UART2_BAUD    921600
#define UART2_TX_PIN  17
#define UART2_RX_PIN  16

// ── Cabeceras de protocolo ──────────────────────────────────
#define PC_HDR_A   0xFF
#define PC_HDR_B   0xFE
#define ESP_HDR_A  0xAA
#define ESP_HDR_B  0x55

// ── Paquete comprimido (heap, pasado por puntero en cola) ───
struct CoeffPacket {
    int16_t  original[N];        // Muestras sin comprimir (para MSE en receptor)
    uint16_t numCoeffs;          // K coeficientes seleccionados (≤ N/2 + 1)
    uint16_t indices[N / 2 + 1]; // Índice de bin (solo mitad positiva del espectro)
    float    re[N / 2 + 1];      // Parte real del coeficiente
    float    im[N / 2 + 1];      // Parte imaginaria del coeficiente
};

// ── Globales de FFT (estáticos, no en stack de tarea) ───────
static double     vReal[N];
static double     vImag[N];
static arduinoFFT FFT;

// ── Cola de punteros entre tareas ───────────────────────────
static QueueHandle_t txQueue;

// ── Estructura auxiliar para ordenar bins por magnitud ──────
struct BinMag {
    double   mag;
    uint16_t idx;
};

// ────────────────────────────────────────────────────────────
//  Tarea Core 0: recepción de audio desde PC + FFT + compresión
// ────────────────────────────────────────────────────────────
void taskFFTCompress(void* /*param*/) {
    static int16_t samples[N];
    uint8_t* rawBytes = reinterpret_cast<uint8_t*>(samples);

    for (;;) {
        // 1. Esperar cabecera del PC [0xFF][0xFE]
        for (;;) {
            while (Serial.available() < 1) taskYIELD();
            if (Serial.read() != PC_HDR_A) continue;
            while (Serial.available() < 1) taskYIELD();
            if (Serial.peek() == PC_HDR_B) { Serial.read(); break; }
        }

        // 2. Leer N muestras int16_t (N×2 bytes, little-endian)
        size_t needed = (size_t)N * sizeof(int16_t);
        size_t got    = 0;
        while (got < needed) {
            if (Serial.available()) rawBytes[got++] = Serial.read();
            else taskYIELD();
        }

        // 3. Copiar a buffers FFT
        for (int i = 0; i < N; i++) {
            vReal[i] = static_cast<double>(samples[i]);
            vImag[i] = 0.0;
        }

        // 4. FFT (sin ventana para que la IFFT reconstruya sin atenuación de bordes)
        FFT.Compute(vReal, vImag, N, FFT_FORWARD);

        // 5. Calcular energía total usando solo la mitad positiva del espectro
        //    (el espectro de señal real es simétrico: E_total = |X[0]|² + 2·Σ|X[k]|² + |X[N/2]|²)
        static BinMag posBins[N / 2 + 1];
        double totalEnergy = 0.0;

        for (int i = 0; i <= N / 2; i++) {
            posBins[i].idx = static_cast<uint16_t>(i);
            posBins[i].mag = sqrt(vReal[i] * vReal[i] + vImag[i] * vImag[i]);
            double e = posBins[i].mag * posBins[i].mag;
            // DC (i=0) y Nyquist (i=N/2) cuentan una vez; los demás cuentan doble
            totalEnergy += (i == 0 || i == N / 2) ? e : 2.0 * e;
        }

        // 6. Ordenar por magnitud descendente
        std::sort(posBins, posBins + N / 2 + 1,
                  [](const BinMag& a, const BinMag& b) { return a.mag > b.mag; });

        // 7. Seleccionar mínimo K bins que acumulen >= ENERGY_THRESHOLD
        CoeffPacket* pkt = new CoeffPacket();
        memcpy(pkt->original, samples, N * sizeof(int16_t));
        pkt->numCoeffs = 0;

        double energyAccum = 0.0;
        for (int i = 0; i <= N / 2; i++) {
            uint16_t idx = posBins[i].idx;
            pkt->indices[pkt->numCoeffs] = idx;
            pkt->re[pkt->numCoeffs]      = static_cast<float>(vReal[idx]);
            pkt->im[pkt->numCoeffs]      = static_cast<float>(vImag[idx]);
            pkt->numCoeffs++;

            double e = posBins[i].mag * posBins[i].mag;
            energyAccum += (idx == 0 || idx == N / 2) ? e : 2.0 * e;
            if (energyAccum / totalEnergy >= ENERGY_THRESHOLD) break;
        }

        // 8. Debug por USB-Serial
        Serial.print("[TX] K=");
        Serial.print(pkt->numCoeffs);
        Serial.print("/");
        Serial.print(N / 2 + 1);
        Serial.print("  Energia=");
        Serial.print(energyAccum / totalEnergy * 100.0, 1);
        Serial.println("%");

        // 9. Encolar paquete para transmisión (descarta si cola llena)
        if (xQueueSend(txQueue, &pkt, pdMS_TO_TICKS(200)) != pdTRUE) {
            delete pkt;
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Tarea Core 1: transmisión binaria via UART2
// ────────────────────────────────────────────────────────────
void taskTransmit(void* /*param*/) {
    CoeffPacket* pkt;

    for (;;) {
        if (xQueueReceive(txQueue, &pkt, portMAX_DELAY) != pdTRUE) continue;

        uint8_t  crc = 0;
        uint16_t blockSize = N;

        // Función local: escribe byte y acumula CRC
        auto wb = [&](uint8_t b) {
            Serial2.write(b);
            crc ^= b;
        };
        auto wbuf = [&](const void* data, size_t len) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
            for (size_t i = 0; i < len; i++) wb(p[i]);
        };

        // Cabecera (fuera del CRC para sincronización del receptor)
        Serial2.write(ESP_HDR_A);
        Serial2.write(ESP_HDR_B);

        // Contenido (todo dentro del CRC)
        wbuf(&blockSize,         sizeof(blockSize));
        wbuf(pkt->original,      N * sizeof(int16_t));
        wbuf(&pkt->numCoeffs,    sizeof(pkt->numCoeffs));
        for (uint16_t k = 0; k < pkt->numCoeffs; k++) {
            wbuf(&pkt->indices[k], sizeof(uint16_t));
            wbuf(&pkt->re[k],      sizeof(float));
            wbuf(&pkt->im[k],      sizeof(float));
        }

        Serial2.write(crc);
        Serial2.flush();

        delete pkt;
    }
}

// ────────────────────────────────────────────────────────────
//  Setup
// ────────────────────────────────────────────────────────────
void setup() {
    // USB-Serial: recibe audio del PC (script Python)
    Serial.begin(115200);

    // UART2: envía coeficientes al receptor ESP32 #2
    Serial2.begin(UART2_BAUD, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);

    Serial.println("=== ESP32 Transmisor FFT ===");
    Serial.print("N="); Serial.print(N);
    Serial.print("  Fs="); Serial.print(SAMPLE_RATE);
    Serial.print(" Hz  Umbral="); Serial.print(ENERGY_THRESHOLD * 100); Serial.println("%");
    Serial.println("Esperando bloques de audio del PC...");

    // Cola de hasta 4 paquetes (punteros)
    txQueue = xQueueCreate(4, sizeof(CoeffPacket*));

    // Tarea FFT + compresión → Core 0
    xTaskCreatePinnedToCore(taskFFTCompress, "FFTCompress",
                            /* stack */ 8192, NULL, /* prio */ 2, NULL, /* core */ 0);

    // Tarea transmisión → Core 1
    xTaskCreatePinnedToCore(taskTransmit, "Transmit",
                            /* stack */ 4096, NULL, /* prio */ 1, NULL, /* core */ 1);
}

void loop() {
    // Toda la lógica está en las tareas FreeRTOS
    vTaskDelay(portMAX_DELAY);
}
