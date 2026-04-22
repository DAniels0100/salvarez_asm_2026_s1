// ============================================================
//  fft_transmitter.ino  —  ESP32 #1: Transmisor (modo preload)
// ============================================================
//  Flujo:
//    1. Recibe TODO el audio del PC por Serial de una vez
//    2. Lo guarda en RAM
//    3. Calcula FFT + compresion bloque a bloque sin presion de tiempo
//    4. Transmite los coeficientes al ESP32 #2 por UART2
//
//  Protocolo PC -> ESP32 (USB-Serial, 115200 baud):
//    [0xAB][0xCD]          : inicio de transferencia
//    num_blocks : uint16_t : cantidad de bloques
//    data       : num_blocks * N * 2 bytes (int16 little-endian)
//    [0xEF][0x01]          : fin de transferencia
//
//  Protocolo ESP32 -> ESP32 (UART2, 921600 baud):
//    [0xAA][0x55]
//    N            : uint16_t
//    original[N]  : int16_t x N
//    K            : uint16_t
//    K x (index:u16, re:f32, im:f32)
//    crc          : uint8_t
//
//  Conexion hardware:
//    ESP32 #1  GPIO 17 (TX2) ---> GPIO 16 (RX2) ESP32 #2
//    ESP32 #1  GND           ---> GND            ESP32 #2
//
//  Libreria requerida: arduinoFFT v2.x
// ============================================================

#include <Arduino.h>
#include <arduinoFFT.h>
#include <algorithm>

// -- Parametros ---------------------------------------------
#define N                  256
#define SAMPLE_RATE        8000.0
#define ENERGY_THRESHOLD   0.95f
#define MAX_BLOCKS         187    // 187 x 256 ~ 6 segundos


// -- UART2 --------------------------------------------------
#define UART2_BAUD 57600
#define UART2_TX_PIN  17
#define UART2_RX_PIN  16

// -- Cabeceras de protocolo ---------------------------------
#define PC_START_A  0xAB
#define PC_START_B  0xCD
#define PC_END_A    0xEF
#define PC_END_B    0x01
#define ESP_HDR_A   0xAA
#define ESP_HDR_B   0x55

// -- Puntero al buffer de audio (heap) --------------------
// Buffer en heap — asignado con malloc en setup()
static int16_t* audioBuffer = nullptr;
static uint16_t totalBlocks = 0;

// -- Buffers FFT --------------------------------------------
static double vReal[N];
static double vImag[N];
static ArduinoFFT<double> FFT(vReal, vImag, N, SAMPLE_RATE);

// -- Estructura para ordenar bins ---------------------------
struct BinMag {
    double   mag;
    uint16_t idx;
};
static BinMag posBins[N / 2 + 1];

// -- Estado de la maquina -----------------------------------
enum State { WAITING, RECEIVING, PROCESSING, DONE };
static volatile State state = WAITING;

// ────────────────────────────────────────────────────────────
//  Recibir todo el audio del PC
// ────────────────────────────────────────────────────────────
bool receiveAudio() {
    Serial.println("[TX] Esperando inicio de transferencia...");

    // Esperar cabecera [0xAB][0xCD]
    while (true) {
        while (Serial.available() < 1) delay(1);
        if (Serial.read() != PC_START_A) continue;
        while (Serial.available() < 1) delay(1);
        if (Serial.read() == PC_START_B) break;
    }

    // Leer numero de bloques
    while (Serial.available() < 2) delay(1);
    uint8_t lo = Serial.read();
    uint8_t hi = Serial.read();
    totalBlocks = (uint16_t)(lo | (hi << 8));

    if (totalBlocks > MAX_BLOCKS) {
        Serial.print("[TX] ERROR: demasiados bloques: ");
        Serial.print(totalBlocks);
        Serial.print(" max=");
        Serial.println(MAX_BLOCKS);
        return false;
    }

    Serial.print("[TX] Recibiendo ");
    Serial.print(totalBlocks);
    Serial.println(" bloques...");

    // Recibir todos los datos
    size_t totalBytes = (size_t)totalBlocks * N * sizeof(int16_t);
    uint8_t* dst = reinterpret_cast<uint8_t*>(audioBuffer);
    size_t received = 0;

    unsigned long lastActivity = millis();
    while (received < totalBytes) {
        if (Serial.available()) {
            dst[received++] = Serial.read();
            lastActivity = millis();
            // Progreso cada 10%
            if (received % (totalBytes / 10) == 0) {
                Serial.print("[TX] Recibido: ");
                Serial.print(received * 100 / totalBytes);
                Serial.println("%");
            }
        } else {
            // Timeout de 5 segundos sin datos
            if (millis() - lastActivity > 5000) {
                Serial.println("[TX] ERROR: timeout esperando datos");
                return false;
            }
            delay(1);
        }
    }

    // Verificar fin de transferencia [0xEF][0x01]
    while (Serial.available() < 2) delay(10);
    uint8_t ea = Serial.read();
    uint8_t eb = Serial.read();
    if (ea != PC_END_A || eb != PC_END_B) {
        Serial.println("[TX] WARN: fin de transferencia incorrecto, continuando...");
    }

    Serial.println("[TX] Audio recibido completamente.");
    return true;
}

// ────────────────────────────────────────────────────────────
//  Procesar y transmitir bloque por bloque
// ────────────────────────────────────────────────────────────
void processAndTransmit() {
    Serial.print("[TX] Procesando y transmitiendo ");
    Serial.print(totalBlocks);
    Serial.println(" bloques...");

    for (uint16_t blk = 0; blk < totalBlocks; blk++) {
        int16_t* samples = &audioBuffer[blk * N];

        // 1. Copiar a buffers FFT
        for (int i = 0; i < N; i++) {
            vReal[i] = static_cast<double>(samples[i]);
            vImag[i] = 0.0;
        }

        // 2. FFT
        FFT.compute(FFTDirection::Forward);

        // 3. Calcular energia total
        double totalEnergy = 0.0;
        for (int i = 0; i <= N / 2; i++) {
            posBins[i].idx = static_cast<uint16_t>(i);
            posBins[i].mag = sqrt(vReal[i]*vReal[i] + vImag[i]*vImag[i]);
            double e = posBins[i].mag * posBins[i].mag;
            totalEnergy += (i == 0 || i == N / 2) ? e : 2.0 * e;
        }

        // 4. Ordenar bins por magnitud
        std::sort(posBins, posBins + N / 2 + 1,
                  [](const BinMag& a, const BinMag& b) { return a.mag > b.mag; });

        // 5. Seleccionar K bins con >= ENERGY_THRESHOLD
        uint16_t numCoeffs = 0;
        uint16_t indices[N / 2 + 1];
        float    re[N / 2 + 1];
        float    im[N / 2 + 1];

        double energyAccum = 0.0;
        for (int i = 0; i <= N / 2; i++) {
            uint16_t idx = posBins[i].idx;
            indices[numCoeffs] = idx;
            re[numCoeffs]      = static_cast<float>(vReal[idx]);
            im[numCoeffs]      = static_cast<float>(vImag[idx]);
            numCoeffs++;

            double e = posBins[i].mag * posBins[i].mag;
            energyAccum += (idx == 0 || idx == N / 2) ? e : 2.0 * e;
            if (energyAccum / totalEnergy >= ENERGY_THRESHOLD) break;
        }

        // 6. Debug cada 10 bloques
        if (blk % 10 == 0) {
            Serial.print("[TX] blk="); Serial.print(blk);
            Serial.print(" K=");       Serial.print(numCoeffs);
            Serial.print("/129  E=");  Serial.print(energyAccum/totalEnergy*100.0, 1);
            Serial.println("%");
        }

        // 7. Transmitir por UART2 (VERSIÓN CORREGIDA)

        uint8_t buffer[4096];
        int idx = 0;

        auto wb = [&](uint8_t b) {
            buffer[idx++] = b;
        };
        auto wbuf = [&](const void* data, size_t len) {
            memcpy(&buffer[idx], data, len);
            idx += len;
        };

        // Header
        wb(ESP_HDR_A);
        wb(ESP_HDR_B);

        // Placeholder para payloadSize
        int sizeIndex = idx;
        idx += 2;

        // Payload real
        uint16_t blockSize = N;
        wbuf(&blockSize, sizeof(blockSize));
        wbuf(samples, N * sizeof(int16_t));
        wbuf(&numCoeffs, sizeof(numCoeffs));

        for (uint16_t k = 0; k < numCoeffs; k++) {
            wbuf(&indices[k], sizeof(uint16_t));
            wbuf(&re[k], sizeof(float));
            wbuf(&im[k], sizeof(float));
        }

        // calcular tamaño del payload
        uint16_t payloadSize = idx - (sizeIndex + 2);
        buffer[sizeIndex]     = payloadSize & 0xFF;
        buffer[sizeIndex + 1] = payloadSize >> 8;

        // CRC sobre TODO el payload
        uint8_t crc = 0;
        for (int i = sizeIndex + 2; i < idx; i++) {
            crc ^= buffer[i];
        }

        wb(crc);

        // Enviar todo de una vez
        Serial2.write(buffer, idx);
        Serial2.flush();

        // Pausa entre bloques para que el receptor pueda procesar
        // Un bloque dura 32ms a 8kHz, damos ese mismo tiempo
        delay(80);   // tiempo para que receptor procese a 115200 baud
    }

    Serial.println("[TX] Transmision completada.");
}

// ────────────────────────────────────────────────────────────
//  Setup
// ────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial2.begin(UART2_BAUD, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);

    // Asignar buffer en heap
    audioBuffer = (int16_t*)malloc(MAX_BLOCKS * N * sizeof(int16_t));
    if (!audioBuffer) {
        Serial.println("[TX] ERROR FATAL: no hay RAM para el buffer de audio");
        while(true) delay(1000);
    }

    Serial.println("=== ESP32 Transmisor FFT (modo preload) ===");
    Serial.print("N="); Serial.print(N);
    Serial.print("  Fs="); Serial.print((int)SAMPLE_RATE);
    Serial.print(" Hz  Umbral="); Serial.print(ENERGY_THRESHOLD * 100);
    Serial.print("%  MaxBloques="); Serial.println(MAX_BLOCKS);
    Serial.println("Libre RAM: " + String(ESP.getFreeHeap()) + " bytes");
}

// ────────────────────────────────────────────────────────────
//  Loop
// ────────────────────────────────────────────────────────────
void loop() {
    if (receiveAudio()) {
        processAndTransmit();
        Serial.println("[TX] Listo. Esperando nuevo audio...");
    } else {
        Serial.println("[TX] Error en recepcion. Reintentando...");
        delay(1000);
    }
}
