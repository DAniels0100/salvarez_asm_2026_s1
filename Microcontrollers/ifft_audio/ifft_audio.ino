// ============================================================
//  ifft_audio.ino  —  ESP32 #2: Receptor (PC como puente)
// ============================================================
//  Flujo FIFO completo:
//    1. Calcula cuantos bloques caben segun heap libre
//    2. Recibe paquetes, hace IFFT por bloque y acumula MSE
//    3. Guarda audio como int8_t (1 byte/muestra = resolucion del DAC)
//    4. Muestra MSE global en LCD
//    5. Reproduce el audio completo sin cortes
//
//  El buffer int8_t usa 4x menos RAM que float y es sin perdidas
//  para el DAC de 8 bits del ESP32 (dacWrite acepta uint8_t).
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

// -- Parametros fijos ----------------------------------------
#define N                   256
#define SAMPLE_RATE         8000.0
#define MAX_BLOCKS          187      // techo global; heap limita el real
#define INTER_PKT_TIMEOUT   3000u    // timeout entre paquetes consecutivos

// -- Pines ---------------------------------------------------
#define SPEAKER_PIN  25

// -- Cabeceras de protocolo ----------------------------------
#define RX_HDR_A  0xAA
#define RX_HDR_B  0x55

// -- LCD 20x4 I2C --------------------------------------------
static LiquidCrystal_I2C lcd(0x27, 20, 4);

// -- Paquete recibido (temporal, uno a la vez en heap) -------
struct RecvPacket {
    int16_t  original[N];
    uint16_t numCoeffs;
    uint16_t indices[N / 2 + 1];
    float    re[N / 2 + 1];
    float    im[N / 2 + 1];
};

// -- Buffer de audio reconstruido ----------------------------
// int8_t normalizado POR BLOQUE: cada bloque usa su propio maximo
// para aprovechar los 8 bits completos. blockMaxes[] guarda la escala
// original para restaurar amplitudes relativas en playback.
static int8_t* reconBuffer = nullptr;
static float*  blockMaxes  = nullptr;  // max IFFT por bloque
static int     allocBlocks = 0;

// -- FFT: buffers en BSS (no consumen heap) ------------------
static double fftRe[N], fftIm[N];
static ArduinoFFT<double> FFT(fftRe, fftIm, N, SAMPLE_RATE);

// ----------------------------------------------------------
static void reconstructSpectrum(const RecvPacket* pkt) {
    memset(fftRe, 0, N * sizeof(double));
    memset(fftIm, 0, N * sizeof(double));
    for (uint16_t k = 0; k < pkt->numCoeffs; k++) {
        uint16_t idx = pkt->indices[k];
        if (idx > N / 2) continue;
        fftRe[idx] = (double)pkt->re[k];
        fftIm[idx] = (double)pkt->im[k];
        if (idx > 0 && idx < N / 2) {
            fftRe[N - idx] =  (double)pkt->re[k];
            fftIm[N - idx] = -(double)pkt->im[k];
        }
    }
}

// ----------------------------------------------------------
// Retorna true solo si se leyo un paquete valido con CRC correcto.
// Retorna false unicamente si paso timeoutMs sin encontrar ningun header.
// En caso de CRC incorrecto o error de lectura, reintenta el siguiente header.
static bool receivePacket(RecvPacket* pkt, unsigned long timeoutMs) {
    unsigned long t0 = millis();

    for (;;) {
        // --- Buscar header [0xAA][0x55] con timeout global ---
        bool gotHeader = false;
        for (;;) {
            if (millis() - t0 > timeoutMs) return false;
            if (!Serial.available()) { taskYIELD(); continue; }
            if (Serial.read() != RX_HDR_A) continue;
            unsigned long t1 = millis();
            while (!Serial.available()) {
                if (millis() - t1 > 200) break;
                taskYIELD();
            }
            if (Serial.available() && Serial.peek() == RX_HDR_B) {
                Serial.read(); gotHeader = true; break;
            }
        }
        if (!gotHeader) return false;

        // --- Leer cuerpo del paquete ---
        uint8_t crc = 0;
        bool ok = true;

        auto rb1 = [&](uint8_t* dst, size_t len) {
            if (!ok) return;
            for (size_t i = 0; i < len; i++) {
                unsigned long t = millis();
                while (!Serial.available()) {
                    if (millis() - t > 500) { ok = false; return; }
                    taskYIELD();
                }
                dst[i] = Serial.read();
                crc ^= dst[i];
            }
        };

        uint16_t blockN = 0;
        rb1((uint8_t*)&blockN, 2);
        if (!ok || blockN != N) continue;   // reintenta siguiente header

        rb1((uint8_t*)pkt->original, N * sizeof(int16_t));
        if (!ok) continue;

        rb1((uint8_t*)&pkt->numCoeffs, sizeof(pkt->numCoeffs));
        if (!ok || pkt->numCoeffs > N / 2 + 1) continue;

        for (uint16_t k = 0; k < pkt->numCoeffs && ok; k++) {
            rb1((uint8_t*)&pkt->indices[k], sizeof(uint16_t));
            rb1((uint8_t*)&pkt->re[k],      sizeof(float));
            rb1((uint8_t*)&pkt->im[k],      sizeof(float));
        }
        if (!ok) continue;

        // Leer CRC (no acumular)
        unsigned long tc = millis();
        while (!Serial.available()) {
            if (millis() - tc > 500) { ok = false; break; }
            taskYIELD();
        }
        if (!ok) continue;

        uint8_t rxCrc = Serial.read();
        if (rxCrc != crc) {
            Serial.println("[WARN] CRC fail, reintentando...");
            continue;   // reintenta siguiente header, no rompe el loop externo
        }

        return true;   // paquete valido
    }
}

// ----------------------------------------------------------
// Procesa un paquete: IFFT, acumula MSE y guarda en reconBuffer
// ----------------------------------------------------------
static void processBlock(const RecvPacket* pkt, int blockIdx,
                          double& mse, double& energyOrig,
                          double& energyRecon, double& energyErr) {
    reconstructSpectrum(pkt);
    FFT.compute(FFTDirection::Reverse);

    // Acumular MSE y encontrar maximo del bloque
    double blockMax = 1.0;
    for (int i = 0; i < N; i++) {
        double orig  = (double)pkt->original[i];
        double recon = fftRe[i];
        double err   = orig - recon;
        mse         += err  * err;
        energyOrig  += orig * orig;
        energyRecon += recon * recon;
        energyErr   += err  * err;
        if (fabs(recon) > blockMax) blockMax = fabs(recon);
    }

    // Guardar escala del bloque para restaurar amplitudes en playback
    blockMaxes[blockIdx] = (float)blockMax;

    // Normalizar al rango completo de int8_t usando el maximo del bloque.
    // Esto evita que bloques quietos se cuanticen a cero (problema con /256 fijo).
    for (int i = 0; i < N; i++) {
        int v = (int)round(fftRe[i] * 127.0 / blockMax);
        reconBuffer[blockIdx * N + i] = (int8_t)constrain(v, -127, 127);
    }

    // Diagnostico cada 20 bloques
    if (blockIdx % 20 == 0) {
        Serial.print("  [BLK "); Serial.print(blockIdx);
        Serial.print("] K="); Serial.print(pkt->numCoeffs);
        Serial.print(" orig[0]="); Serial.print(pkt->original[0]);
        Serial.print(" ifft_max="); Serial.print(blockMax, 1);
        Serial.print(" buf[0]="); Serial.println((int)reconBuffer[blockIdx * N]);
    }
}

// ----------------------------------------------------------
// Beep de confirmacion: tono cuadrado ~1 kHz por 150 ms
// ----------------------------------------------------------
static void beep() {
    for (int i = 0; i < 150; i++) {           // 150 ciclos
        dacWrite(SPEAKER_PIN, 220);            // alto
        delayMicroseconds(500);
        dacWrite(SPEAKER_PIN, 36);             // bajo
        delayMicroseconds(500);
    }
    dacWrite(SPEAKER_PIN, 128);                // silencio
}

// ----------------------------------------------------------
void mainTask(void* /*param*/) {

    // ---- Calcular cuantos bloques caben en heap libre ----
    int freeHeap = (int)ESP.getFreeHeap();
    allocBlocks  = (freeHeap - 6144) / N;
    allocBlocks  = constrain(allocBlocks, 1, MAX_BLOCKS);

    float durSec = (float)(allocBlocks * N) / 8000.0f;

    Serial.print("[INFO] Heap libre:  "); Serial.print(freeHeap); Serial.println(" B");
    Serial.print("[INFO] Bloques max: "); Serial.println(allocBlocks);
    Serial.print("[INFO] Audio max:   "); Serial.print(durSec, 1); Serial.println(" s");

    // ---- Asignar buffers ----
    reconBuffer = (int8_t*)malloc((size_t)allocBlocks * N);
    blockMaxes  = (float*)malloc((size_t)allocBlocks * sizeof(float));
    RecvPacket* pkt = (RecvPacket*)malloc(sizeof(RecvPacket));

    if (!reconBuffer || !blockMaxes || !pkt) {
        lcd.clear();
        lcd.setCursor(0, 0); lcd.print("ERROR: sin RAM");
        lcd.setCursor(0, 1); lcd.print("Heap: "); lcd.print(freeHeap); lcd.print("B");
        Serial.println("[ERROR] malloc fallo");
        vTaskDelete(NULL);
        return;
    }

    double mse = 0.0, energyOrig = 0.0, energyRecon = 0.0, energyErr = 0.0;
    int    blockIdx = 0;

    // =================================================================
    //  FASE 1: Esperar el primer paquete (bloqueo indefinido)
    // =================================================================
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("ESP32 #2 Listo");
    lcd.setCursor(0, 1); lcd.print("Max "); lcd.print(durSec, 1); lcd.print("s");
    lcd.setCursor(0, 2); lcd.print("Esperando script...");
    Serial.println("[FASE 1] Esperando primer paquete...");

    while (!receivePacket(pkt, INTER_PKT_TIMEOUT)) {
        // bloqueo hasta que llegue el primer paquete valido
    }

    Serial.println("[FASE 1] Primer paquete recibido. BEEP.");

    // ---- BEEP de confirmacion ----
    lcd.setCursor(0, 2); lcd.print("Primer paquete OK!  ");
    beep();

    // Procesar primer bloque
    processBlock(pkt, blockIdx, mse, energyOrig, energyRecon, energyErr);
    blockIdx++;

    // =================================================================
    //  FASE 2: Recibir el resto de paquetes + IFFT + MSE acumulado
    // =================================================================
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("Recibiendo audio");
    lcd.setCursor(0, 1); lcd.print("Max "); lcd.print(durSec, 1); lcd.print("s");
    Serial.println("[FASE 2] Recibiendo resto de paquetes...");

    while (blockIdx < allocBlocks) {
        if (!receivePacket(pkt, INTER_PKT_TIMEOUT)) break;   // fin de transmision
        processBlock(pkt, blockIdx, mse, energyOrig, energyRecon, energyErr);
        blockIdx++;

        if (blockIdx % 5 == 0 || blockIdx == allocBlocks) {
            lcd.setCursor(0, 2);
            lcd.print("Bloque ");
            lcd.print(blockIdx);
            lcd.print("/");
            lcd.print(allocBlocks);
            lcd.print("    ");
        }
    }

    free(pkt);
    const int totalSamples = blockIdx * N;
    Serial.print("[FASE 2] Fin. Bloques recibidos: "); Serial.println(blockIdx);

    if (blockIdx == 0) {
        lcd.clear();
        lcd.setCursor(0, 0); lcd.print("Sin paquetes.");
        free(reconBuffer); reconBuffer = nullptr;
        vTaskDelete(NULL); return;
    }

    // =================================================================
    //  FASE 3: Calculo y despliegue de metricas globales
    // =================================================================
    mse /= totalSamples;
    double energyPct = (energyOrig > 0.0) ? (energyRecon / energyOrig) * 100.0 : 0.0;
    double snr       = (energyErr  > 1e-9) ? 10.0 * log10(energyOrig / energyErr) : 99.9;

    Serial.println("[FASE 3] Metricas globales:");
    Serial.print("  MSE:     "); Serial.println(mse, 4);
    Serial.print("  SNR dB:  "); Serial.println(snr, 2);
    Serial.print("  Energia: "); Serial.print(energyPct, 2); Serial.println(" %");

    // Diagnostico: mostrar primeros valores del buffer para verificar contenido
    Serial.print("  reconBuffer[0..7]: ");
    for (int i = 0; i < 8; i++) {
        Serial.print((int)reconBuffer[i]); Serial.print(" ");
    }
    Serial.println();

    // Calcular rango real del buffer
    int8_t bufMin = 127, bufMax = -127;
    for (int i = 0; i < totalSamples; i++) {
        if (reconBuffer[i] < bufMin) bufMin = reconBuffer[i];
        if (reconBuffer[i] > bufMax) bufMax = reconBuffer[i];
    }
    Serial.print("  Buffer rango: ["); Serial.print((int)bufMin);
    Serial.print(", "); Serial.print((int)bufMax); Serial.println("]");

    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("MSE: ");     lcd.print(mse, 2);
    lcd.setCursor(0, 1); lcd.print("Energia: "); lcd.print(energyPct, 1); lcd.print("%");
    lcd.setCursor(0, 2); lcd.print("SNR: ");     lcd.print(snr, 1);       lcd.print(" dB");
    lcd.setCursor(0, 3); lcd.print("Bloques: "); lcd.print(blockIdx);

    vTaskDelay(pdMS_TO_TICKS(2500));

    // =================================================================
    //  FASE 4: Reproduccion del audio reconstruido completo
    // =================================================================
    lcd.setCursor(0, 3); lcd.print("REPRODUCIENDO...    ");
    Serial.println("[FASE 4] Iniciando reproduccion...");

    // Referencia de amplitud: usa 1.5x la media de blockMaxes en vez del maximo.
    // Esto evita que un bloque muy fuerte (outlier) baje el volumen de todo lo demas.
    // Bloques por encima del umbral simplemente se recortan (clipping aceptable).
    float globalMax = 1.0f;
    float sumBM = 0.0f;
    for (int b = 0; b < blockIdx; b++) {
        if (blockMaxes[b] > globalMax) globalMax = blockMaxes[b];
        sumBM += blockMaxes[b];
    }
    float avgBM  = sumBM / blockIdx;
    float refMax = avgBM * 1.5f;
    if (refMax < 1.0f) refMax = 1.0f;

    Serial.print("  blockMaxes avg="); Serial.print(avgBM, 1);
    Serial.print(" max="); Serial.print(globalMax, 1);
    Serial.print(" refMax="); Serial.println(refMax, 1);

    // Tono de prueba 0.3 s (~880 Hz)
    Serial.println("[FASE 4] Tono de prueba...");
    for (int i = 0; i < 2400; i++) {
        dacWrite(SPEAKER_PIN, ((i % 9) < 5) ? 210 : 46);
        delayMicroseconds(125);
    }
    dacWrite(SPEAKER_PIN, 128);
    vTaskDelay(pdMS_TO_TICKS(150));

    // Reproduccion bloque a bloque restaurando amplitudes relativas.
    // Cada bloque b fue normalizado a su propio max (almacenado en blockMaxes[b]).
    // factor[b] = blockMaxes[b] / globalMax → bloque mas fuerte llena el DAC,
    // bloques quietos suenan quietos (comportamiento correcto).
    Serial.println("[FASE 4] Reproduciendo audio reconstruido...");
    for (int b = 0; b < blockIdx; b++) {
        // scale > 1 para bloques mas fuertes que la media → clipping en constrain()
        float scale = blockMaxes[b] / refMax;
        int   base  = b * N;
        for (int i = 0; i < N; i++) {
            float s  = (float)reconBuffer[base + i] * scale;
            int   dv = (int)(s + 128.5f);
            dacWrite(SPEAKER_PIN, (uint8_t)constrain(dv, 0, 255));
            delayMicroseconds(125);
        }
        // Ceder al idle task cada 8 bloques (~256 ms) para evitar watchdog
        if ((b & 0x7) == 0x7) vTaskDelay(pdMS_TO_TICKS(1));
    }

    dacWrite(SPEAKER_PIN, 128);
    free(reconBuffer); reconBuffer = nullptr;
    free(blockMaxes);  blockMaxes  = nullptr;
    Serial.println("[FASE 4] Reproduccion completa.");

    // =================================================================
    //  FASE 5: Metricas finales permanentes en LCD
    // =================================================================
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("MSE: ");     lcd.print(mse, 2);
    lcd.setCursor(0, 1); lcd.print("Energia: "); lcd.print(energyPct, 1); lcd.print("%");
    lcd.setCursor(0, 2); lcd.print("SNR: ");     lcd.print(snr, 1);       lcd.print(" dB");
    lcd.setCursor(0, 3); lcd.print("Listo. "); lcd.print(blockIdx); lcd.print("blks");

    vTaskDelete(NULL);
}

// ----------------------------------------------------------
void setup() {
    Serial.begin(115200);

    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0); lcd.print("ESP32 #2 Receptor");
    lcd.setCursor(0, 1); lcd.print("Esperando datos...");

    dacWrite(SPEAKER_PIN, 128);

    xTaskCreatePinnedToCore(mainTask, "Main", 12288, NULL, 2, NULL, 1);
}

void loop() {
    vTaskDelay(portMAX_DELAY);
}
