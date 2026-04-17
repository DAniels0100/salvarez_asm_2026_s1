"""
send_audio.py — Envía un archivo .wav al ESP32 Transmisor por Serial.

Convierte automáticamente a mono y remuestrea a 8000 Hz si es necesario.
Empaqueta bloques de N=256 muestras int16 con el header [0xFF][0xFE]
que espera el ESP32.

Uso:
    python send_audio.py <archivo.wav> <puerto>  [--loop] [--verbose]

Ejemplos:
    python send_audio.py tono.wav /dev/cu.usbserial-0001        # macOS/Linux
    python send_audio.py tono.wav COM3                           # Windows
    python send_audio.py tono.wav /dev/ttyUSB0 --loop           # repetir
    python send_audio.py tono.wav COM3 --verbose                 # ver métricas del ESP32

Dependencias:
    pip install pyserial numpy scipy
"""

import sys
import time
import argparse
import threading

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sig
import serial

# ── Parámetros — deben coincidir con el .ino ────────────────
N          = 256       # Tamaño 
TARGET_FS  = 8000      # Hz
BAUD_PC    = 115200    # Baud rate USB (Serial del ESP32)
HEADER     = bytes([0xFF, 0xFE])
BLOCK_DUR  = N / TARGET_FS   # 32 ms por bloque


# ────────────────────────────────────────────────────────────
#  Carga y normalización del WAV
# ────────────────────────────────────────────────────────────
def load_wav(path: str) -> np.ndarray:
    """
    Carga un WAV, lo convierte a mono, lo remuestrea a TARGET_FS
    y devuelve un array de int16.
    """
    fs, data = wavfile.read(path)

    # Convertir a float64 con rango [-32768, 32767]
    dtype = data.dtype
    if dtype == np.uint8:               # 8-bit unsigned [0, 255]
        audio = (data.astype(np.float64) - 128.0) * 256.0
    elif dtype == np.int16:             # ya en rango correcto
        audio = data.astype(np.float64)
    elif dtype == np.int32:             # 32-bit → escalar a int16
        audio = data.astype(np.float64) / 65536.0
    elif dtype in (np.float32, np.float64):  # [-1.0, 1.0]
        audio = data.astype(np.float64) * 32767.0
    else:
        raise ValueError(f"Formato de muestra no soportado: {dtype}")

    # Convertir a mono si es estéreo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Remuestrear si la frecuencia de muestreo es distinta
    if fs != TARGET_FS:
        gcd  = np.gcd(int(fs), int(TARGET_FS))
        up   = TARGET_FS // gcd
        down = fs        // gcd
        print(f"  Remuestreando {fs} Hz → {TARGET_FS} Hz (↑{up} ↓{down})...")
        audio = sig.resample_poly(audio, up, down)

    # Recortar y convertir a int16
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio


# ────────────────────────────────────────────────────────────
#  Hilo lector: muestra lo que el ESP32 responde por Serial
# ────────────────────────────────────────────────────────────
def reader_thread(ser: serial.Serial, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"  [ESP32] {line}")
        except Exception:
            break
        time.sleep(0.005)


# ────────────────────────────────────────────────────────────
#  Envío principal
# ────────────────────────────────────────────────────────────
def send_audio(port: str, audio: np.ndarray,
               loop: bool = False, verbose: bool = False):

    total_samples = len(audio)
    num_blocks    = total_samples // N
    total_dur     = total_samples / TARGET_FS

    print(f"\n{'─'*50}")
    print(f"  Puerto   : {port}  ({BAUD_PC} baud)")
    print(f"  Audio    : {total_samples} muestras  "
          f"{total_dur:.2f} s  →  {num_blocks} bloques de {N}")
    print(f"  Cadencia : {BLOCK_DUR*1000:.1f} ms/bloque  "
          f"({TARGET_FS} Hz)")
    print(f"  Loop     : {'sí' if loop else 'no'}")
    print(f"{'─'*50}\n")

    with serial.Serial(port, BAUD_PC, timeout=1) as ser:
        # Esperar reset del ESP32 tras abrir el puerto
        print("Esperando reset del ESP32 (2 s)...")
        time.sleep(2)
        ser.reset_input_buffer()

        stop_event = threading.Event()
        if verbose:
            t = threading.Thread(target=reader_thread,
                                 args=(ser, stop_event), daemon=True)
            t.start()

        iteration = 0
        try:
            while True:
                iteration += 1
                if loop:
                    print(f"\n── Reproducción #{iteration} ──")

                for i in range(num_blocks):
                    t0    = time.perf_counter()
                    block = audio[i * N : (i + 1) * N]

                    # Protocolo: [0xFF][0xFE] + N × int16 (little-endian)
                    ser.write(HEADER)
                    ser.write(block.tobytes())

                    # Indicador de progreso
                    pct = (i + 1) / num_blocks * 100
                    bar = "#" * int(pct / 5)
                    print(f"\r  [{bar:<20}] {pct:5.1f}%  bloque {i+1}/{num_blocks}",
                          end="", flush=True)

                    # Temporizar para mantener cadencia real de audio
                    elapsed = time.perf_counter() - t0
                    wait    = BLOCK_DUR - elapsed
                    if wait > 0:
                        time.sleep(wait)

                print(f"\r  [{'#'*20}] 100.0%  bloque {num_blocks}/{num_blocks}")
                print("  ✓ Audio enviado.")

                if not loop:
                    break

        except KeyboardInterrupt:
            print("\n\n  Interrumpido por el usuario.")
        finally:
            stop_event.set()

    print("Puerto cerrado.\n")


# ────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Envía un .wav al ESP32 Transmisor por Serial."
    )
    parser.add_argument("wav",    help="Ruta al archivo .wav")
    parser.add_argument("puerto", help="Puerto serial (ej. /dev/cu.usbserial-0001 o COM3)")
    parser.add_argument("--loop", action="store_true",
                        help="Repetir el audio en bucle hasta Ctrl+C")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar respuestas del ESP32 por Serial")
    args = parser.parse_args()

    print(f"\nCargando '{args.wav}'...")
    try:
        audio = load_wav(args.wav)
    except FileNotFoundError:
        print(f"Error: no se encontró el archivo '{args.wav}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error al cargar el WAV: {e}")
        sys.exit(1)

    print(f"  OK — {len(audio)} muestras int16 @ {TARGET_FS} Hz")

    send_audio(args.puerto, audio, loop=args.loop, verbose=args.verbose)


if __name__ == "__main__":
    main()
