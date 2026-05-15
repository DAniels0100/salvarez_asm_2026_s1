// ============================================================
//  fft_transmitter.ino  —  ESP32 #1: Transmisor
// ============================================================
//  PC actua como puente. Flujo:
//    1. PC envia el .wav al ESP32 #1 por USB-Serial
//    2. ESP32 #1 guarda en RAM, calcula FFT + compresion
//    3. ESP32 #1 envia los coeficientes comprimidos de vuelta al PC por USB
//    4. PC reenvia esos coeficientes al ESP32 #2 por USB (otro puerto)
//
//  Protocolo PC -> ESP32 (USB, 115200 baud):
//    [0xAB][0xCD]          : inicio de transferencia
//    num_blocks : uint16_t : cantidad de bloques
//    data       : num_blocks * N * 2 bytes (int16 LE)
//    [0xEF][0x01]          : fin de transferencia
//
//  Protocolo ESP32 -> PC (USB, 115200 baud):
//    Lineas de texto (debug) y paquetes binarios intercalados.
//    Los paquetes binarios comienzan con [0xAA][0x55] para que el PC
//    los distinga de los logs de texto.
//
//    Paquete binario:
//      [0xAA][0x55]
//      N            : uint16_t
//      original[N]  : int16_t x N
//      K            : uint16_t
//      K x (index:u16, re:f32, im:f32)
//      crc          : uint8_t
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
#define MAX_BLOCKS         187    // ~6 segundos

// -- Cabeceras de protocolo ---------------------------------
#define PC_START_A  0xAB
#define PC_START_B  0xCD
#define PC_END_A    0xEF
#define PC_END_B    0x01
#define TX_HDR_A    0xAA
#define TX_HDR_B    0x55

// -- Buffer de audio en heap --------------------------------
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

// ----------------------------------------------------------
//  Recibir audio completo del PC
// ----------------------------------------------------------
bool receiveAudio() {
    Serial.println("[TX] Esperando inicio de transferencia...");

    while (true) {
        while (Serial.available() < 1) delay(1);
        if (Serial.read() != PC_START_A) continue;
        while (Serial.available() < 1) delay(1);
        if (Serial.read() == PC_START_B) break;
    }

    while (Serial.available() < 2) delay(1);
    uint8_t lo = Serial.read();
    uint8_t hi = Serial.read();
    totalBlocks = (uint16_t)(lo | (hi << 8));

    if (totalBlocks > MAX_BLOCKS) {
        Serial.print("[TX] ERROR: demasiados bloques: ");
        Serial.println(totalBlocks);
        return false;
    }

    Serial.print("[TX] Recibiendo ");
    Serial.print(totalBlocks);
    Serial.println(" bloques...");

    size_t totalBytes = (size_t)totalBlocks * N * sizeof(int16_t);
    uint8_t* dst = reinterpret_cast<uint8_t*>(audioBuffer);
    size_t received = 0;

    unsigned long lastActivity = millis();
    while (received < totalBytes) {
        if (Serial.available()) {
            dst[received++] = Serial.read();
            lastActivity = millis();
        } else {
            if (millis() - lastActivity > 5000) {
                Serial.println("[TX] ERROR: timeout");
                return false;
            }
            delay(1);
        }
    }

    // Leer footer
    while (Serial.available() < 2) delay(10);
    Serial.read(); Serial.read();

    Serial.println("[TX] Audio recibido completamente.");
    return true;
}

// ----------------------------------------------------------
//  Procesar y enviar coeficientes de vuelta al PC
// ----------------------------------------------------------
void processAndSend() {
    Serial.print("[TX] Procesando ");
    Serial.print(totalBlocks);
    Serial.println(" bloques...");

    // Aviso al PC que vienen paquetes binarios
    Serial.println("[TX] BEGIN_BINARY");
    Serial.flush();
    delay(100);

    for (uint16_t blk = 0; blk < totalBlocks; blk++) {
        int16_t* samples = &audioBuffer[blk * N];

        // 1. FFT
        for (int i = 0; i < N; i++) {
            vReal[i] = static_cast<double>(samples[i]);
            vImag[i] = 0.0;
        }
        FFT.compute(FFTDirection::Forward);

        // 2. Energia total
        double totalEnergy = 0.0;
        for (int i = 0; i <= N / 2; i++) {
            posBins[i].idx = static_cast<uint16_t>(i);
            posBins[i].mag = sqrt(vReal[i]*vReal[i] + vImag[i]*vImag[i]);
            double e = posBins[i].mag * posBins[i].mag;
            totalEnergy += (i == 0 || i == N / 2) ? e : 2.0 * e;
        }

        // 3. Ordenar
        std::sort(posBins, posBins + N / 2 + 1,
                  [](const BinMag& a, const BinMag& b) { return a.mag > b.mag; });

        // 4. Seleccionar K bins
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

        // 5. Enviar paquete binario por USB al PC
        uint8_t crc = 0;
        uint16_t blockSize = N;

        auto wb = [&](uint8_t b) {
            Serial.write(b);
            crc ^= b;
        };
        auto wbuf = [&](const void* data, size_t len) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
            for (size_t i = 0; i < len; i++) wb(p[i]);
        };

        Serial.write(TX_HDR_A);
        Serial.write(TX_HDR_B);
        wbuf(&blockSize,  sizeof(blockSize));
        wbuf(samples,     N * sizeof(int16_t));
        wbuf(&numCoeffs,  sizeof(numCoeffs));
        for (uint16_t k = 0; k < numCoeffs; k++) {
            wbuf(&indices[k], sizeof(uint16_t));
            wbuf(&re[k],      sizeof(float));
            wbuf(&im[k],      sizeof(float));
        }
        Serial.write(crc);
        Serial.flush();

        // Pausa para no saturar el buffer del PC
        delay(10);
    }

    Serial.println("\n[TX] END_BINARY");
    Serial.println("[TX] Transmision completada.");
}

// ----------------------------------------------------------
//  Setup
// ----------------------------------------------------------
void setup() {
    Serial.begin(115200);

    audioBuffer = (int16_t*)malloc(MAX_BLOCKS * N * sizeof(int16_t));
    if (!audioBuffer) {
        Serial.println("[TX] ERROR FATAL: sin RAM para buffer");
        while(true) delay(1000);
    }

    Serial.println("=== ESP32 Transmisor FFT (PC como puente) ===");
    Serial.print("N="); Serial.print(N);
    Serial.print("  Fs="); Serial.print((int)SAMPLE_RATE);
    Serial.print(" Hz  MaxBloques="); Serial.println(MAX_BLOCKS);
    Serial.println("Libre RAM: " + String(ESP.getFreeHeap()));
}

// ----------------------------------------------------------
//  Loop
// ----------------------------------------------------------
void loop() {
    if (receiveAudio()) {
        processAndSend();
        Serial.println("[TX] Listo. Esperando nuevo audio...");
    } else {
        Serial.println("[TX] Error. Reintentando...");
        delay(1000);
    }
}
