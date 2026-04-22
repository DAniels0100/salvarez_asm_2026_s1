"""
send_audio.py — Envia un archivo .wav al ESP32 Transmisor por Serial (modo preload).

El script envia todo el audio de una vez con el protocolo:
  [0xAB][0xCD] + num_blocks (uint16 LE) + datos (N*num_blocks*2 bytes) + [0xEF][0x01]

El ESP32 guarda todo en RAM, hace FFT+compresion y transmite al receptor.

Uso:
    python send_audio.py <archivo.wav> <puerto> [--verbose]

Ejemplos:
    python send_audio.py audio.wav /dev/cu.SLAB_USBtoUART6
    python send_audio.py audio.wav /dev/cu.SLAB_USBtoUART6 --verbose

Dependencias:
    pip install pyserial numpy scipy
"""

import sys
import time
import argparse
import threading
import struct

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sig
import serial

# -- Parametros — deben coincidir con el .ino ---------------
N          = 256
TARGET_FS  = 8000
BAUD_PC    = 115200
MAX_BLOCKS = 187      # igual que en el .ino (6 segundos max)

# Cabeceras del protocolo
START_HEADER = bytes([0xAB, 0xCD])
END_FOOTER   = bytes([0xEF, 0x01])


# -----------------------------------------------------------
#  Carga y normalizacion del WAV
# -----------------------------------------------------------
def load_wav(path: str) -> np.ndarray:
    fs, data = wavfile.read(path)

    dtype = data.dtype
    if dtype == np.uint8:
        audio = (data.astype(np.float64) - 128.0) * 256.0
    elif dtype == np.int16:
        audio = data.astype(np.float64)
    elif dtype == np.int32:
        audio = data.astype(np.float64) / 65536.0
    elif dtype in (np.float32, np.float64):
        audio = data.astype(np.float64) * 32767.0
    else:
        raise ValueError(f"Formato no soportado: {dtype}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if fs != TARGET_FS:
        gcd  = np.gcd(int(fs), int(TARGET_FS))
        up   = TARGET_FS // gcd
        down = fs        // gcd
        print(f"  Remuestreando {fs} Hz -> {TARGET_FS} Hz...")
        audio = sig.resample_poly(audio, up, down)

    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio


# -----------------------------------------------------------
#  Hilo lector: muestra respuestas del ESP32
# -----------------------------------------------------------
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


# -----------------------------------------------------------
#  Envio principal
# -----------------------------------------------------------
def send_audio(port: str, audio: np.ndarray, verbose: bool = False):
    # Calcular bloques — limitar a MAX_BLOCKS
    total_blocks = len(audio) // N
    if total_blocks > MAX_BLOCKS:
        print(f"  WARN: audio tiene {total_blocks} bloques, recortando a {MAX_BLOCKS} ({MAX_BLOCKS*N/TARGET_FS:.1f}s)")
        total_blocks = MAX_BLOCKS
        audio = audio[:total_blocks * N]

    total_bytes = total_blocks * N * 2  # int16 = 2 bytes

    print(f"\n{'─'*50}")
    print(f"  Puerto    : {port}  ({BAUD_PC} baud)")
    print(f"  Bloques   : {total_blocks} x {N} muestras = {total_blocks*N/TARGET_FS:.1f}s")
    print(f"  Datos     : {total_bytes/1024:.1f} KB a transferir")
    print(f"{'─'*50}\n")

    with serial.Serial(port, BAUD_PC, timeout=5) as ser:
        print("Esperando reset del ESP32 (2s)...")
        time.sleep(2)
        ser.reset_input_buffer()

        stop_event = threading.Event()
        if verbose:
            t = threading.Thread(target=reader_thread,
                                 args=(ser, stop_event), daemon=True)
            t.start()

        try:
            # 1. Enviar cabecera de inicio
            print("Enviando cabecera...")
            ser.write(START_HEADER)

            # 2. Enviar numero de bloques (uint16 little-endian)
            ser.write(struct.pack('<H', total_blocks))

            # 3. Enviar todos los datos de audio
            print(f"Enviando {total_bytes/1024:.1f} KB de audio...")
            raw = audio[:total_blocks * N].tobytes()

            chunk_size = 1024
            sent = 0
            t0 = time.perf_counter()
            while sent < len(raw):
                chunk = raw[sent:sent+chunk_size]
                ser.write(chunk)
                sent += len(chunk)

                pct = sent / len(raw) * 100
                bar = "#" * int(pct / 5)
                elapsed = time.perf_counter() - t0
                kbps = (sent / 1024) / elapsed if elapsed > 0 else 0
                print(f"\r  [{bar:<20}] {pct:5.1f}%  {sent//1024}/{len(raw)//1024} KB  {kbps:.0f} KB/s",
                      end="", flush=True)

            print(f"\r  [{'#'*20}] 100.0%  {len(raw)//1024}/{len(raw)//1024} KB")

            # 4. Enviar footer de fin
            ser.write(END_FOOTER)
            ser.flush()

            elapsed = time.perf_counter() - t0
            print(f"  Audio enviado en {elapsed:.1f}s")
            print(f"  Velocidad promedio: {(total_bytes/1024)/elapsed:.0f} KB/s")
            print("\nEsperando que el ESP32 procese y transmita al receptor...")
            print("(Esto puede tomar unos segundos)\n")

            # Esperar respuestas del ESP32 mientras procesa
            if verbose:
                # Dar tiempo al ESP32 para procesar todo
                process_time = total_blocks * 0.035  # ~35ms por bloque
                print(f"  Tiempo estimado de procesamiento: {process_time:.0f}s")
                time.sleep(process_time + 5)
            else:
                time.sleep(10)

        except KeyboardInterrupt:
            print("\n\nInterrumpido por el usuario.")
        finally:
            stop_event.set()

    print("Puerto cerrado.")


# -----------------------------------------------------------
#  Entry point
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Envia un .wav al ESP32 Transmisor (modo preload)."
    )
    parser.add_argument("wav",    help="Ruta al archivo .wav")
    parser.add_argument("puerto", help="Puerto serial")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar respuestas del ESP32")
    args = parser.parse_args()

    print(f"\nCargando '{args.wav}'...")
    try:
        audio = load_wav(args.wav)
    except FileNotFoundError:
        print(f"Error: no se encontro '{args.wav}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error al cargar WAV: {e}")
        sys.exit(1)

    print(f"  OK — {len(audio)} muestras int16 @ {TARGET_FS} Hz ({len(audio)/TARGET_FS:.1f}s)")
    send_audio(args.puerto, audio, verbose=args.verbose)


if __name__ == "__main__":
    main()
