"""
send_audio.py — Envia .wav al ESP32 #1 y actua como puente al ESP32 #2.

Flujo:
  1. Carga el .wav y lo envia al ESP32 #1 por USB
  2. ESP32 #1 hace FFT + compresion y devuelve paquetes binarios al PC
  3. PC lee los paquetes binarios y los reenvia al ESP32 #2 por USB

Uso:
    python send_audio.py <wav> <puerto_tx> <puerto_rx> [--verbose]

Ejemplo:
    python send_audio.py audio.wav /dev/cu.SLAB_USBtoUART6 /dev/cu.SLAB_USBtoUART --verbose

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

# -- Parametros (deben coincidir con los .ino) --------------
N          = 256
TARGET_FS  = 8000
BAUD       = 115200
MAX_BLOCKS = 187

START_HEADER = bytes([0xAB, 0xCD])
END_FOOTER   = bytes([0xEF, 0x01])
PKT_HDR      = bytes([0xAA, 0x55])

# Control frames (PC -> RX) using the same [0xAA][0x55] sync:
#   [AA55][blockN=0x0000][cmd:u8][(totalBlocks:u16 si cmd=1)][crc:u8]
# cmd: 1=START, 2=END


def build_control_frame(cmd: int, total_blocks: int | None = None) -> bytes:
    blockN = 0
    payload = struct.pack('<H', blockN) + struct.pack('<B', cmd)
    if cmd == 1:
        if total_blocks is None:
            raise ValueError('total_blocks requerido para START')
        payload += struct.pack('<H', int(total_blocks) & 0xFFFF)

    crc = 0
    for b in payload:
        crc ^= b
    return PKT_HDR + payload + struct.pack('<B', crc)

# Tamano de cada paquete binario recibido del TX
# 2 (N) + 256*2 (original) + 2 (K) + Kmax*(2+4+4) = variable
# Maximo: 2 + 512 + 2 + 129*10 + 1 = 1807 bytes
# Minimo: 2 + 512 + 2 + 1*10 + 1 = 527 bytes


# -----------------------------------------------------------
#  Carga WAV
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
#  Lee un paquete binario completo del ESP32 #1
# -----------------------------------------------------------
def read_packet(ser_tx: serial.Serial, verbose: bool):
    """
    Lee bytes hasta encontrar [0xAA][0x55], luego el paquete completo.
    Devuelve el paquete completo (con header) o None si timeout.
    Ignora lineas de texto (logs) antes del binario.
    """
    # Buscar sincronizacion [0xAA][0x55]
    # Mientras buscamos, tambien imprimimos logs de texto
    line_buf = bytearray()
    t0 = time.time()

    while True:
        if time.time() - t0 > 30:
            return None  # timeout global

        if ser_tx.in_waiting < 1:
            time.sleep(0.001)
            continue

        b = ser_tx.read(1)
        if not b:
            continue

        # Acumular potencial linea de texto
        line_buf.append(b[0])

        # Si llega '\n', era una linea de log
        if b == b'\n':
            try:
                line = line_buf.decode('utf-8', errors='replace').rstrip()
                if verbose and line:
                    print(f"  [TX] {line}")
            except Exception:
                pass
            line_buf = bytearray()
            continue

        # Si llevamos 2 bytes y coinciden con header, sync encontrada
        if len(line_buf) >= 2 and line_buf[-2] == 0xAA and line_buf[-1] == 0x55:
            break

        # Evitar crecimiento ilimitado del line_buf si no hay newlines
        if len(line_buf) > 4096:
            line_buf = line_buf[-2:]

    # Leer cuerpo del paquete
    # Primero: N (2 bytes) + original (512 bytes) + K (2 bytes)
    header_rest = ser_tx.read(2 + 512 + 2)
    if len(header_rest) < 516:
        return None

    N_val = struct.unpack('<H', header_rest[0:2])[0]
    if N_val != N:
        return None

    K = struct.unpack('<H', header_rest[514:516])[0]
    if K > N // 2 + 1:
        return None

    # Leer K coeficientes (cada uno 10 bytes: u16 + f32 + f32)
    coeffs = ser_tx.read(K * 10)
    if len(coeffs) < K * 10:
        return None

    # Leer CRC
    crc_byte = ser_tx.read(1)
    if len(crc_byte) < 1:
        return None

    # Reconstruir paquete completo a enviar al receptor
    # Formato: [0xAA][0x55] + header_rest + coeffs + crc
    packet = bytes(PKT_HDR) + bytes(header_rest) + bytes(coeffs) + bytes(crc_byte)
    return packet


# -----------------------------------------------------------
#  Hilo lector del RX — muestra lo que el ESP32 #2 imprime
# -----------------------------------------------------------
def rx_reader(ser_rx: serial.Serial, stop_event: threading.Event):
    buf = bytearray()
    while not stop_event.is_set():
        try:
            if ser_rx.in_waiting:
                data = ser_rx.read(ser_rx.in_waiting)
                buf.extend(data)
                # Imprimir lineas completas
                while b'\n' in buf:
                    line, _, buf = buf.partition(b'\n')
                    try:
                        txt = line.decode('utf-8', errors='replace').rstrip()
                        if txt:
                            print(f"  [RX] {txt}")
                    except Exception:
                        pass
        except Exception:
            break
        time.sleep(0.01)


# -----------------------------------------------------------
#  Envio principal
# -----------------------------------------------------------
def run_bridge(port_tx: str, port_rx: str, audio: np.ndarray, verbose: bool):
    total_blocks = len(audio) // N
    if total_blocks > MAX_BLOCKS:
        print(f"  WARN: recortando a {MAX_BLOCKS} bloques ({MAX_BLOCKS*N/TARGET_FS:.1f}s)")
        total_blocks = MAX_BLOCKS
        audio = audio[:total_blocks * N]

    total_bytes = total_blocks * N * 2

    print(f"\n{'─'*50}")
    print(f"  ESP32 #1 (TX): {port_tx}")
    print(f"  ESP32 #2 (RX): {port_rx}")
    print(f"  Bloques:       {total_blocks} ({total_blocks*N/TARGET_FS:.1f}s)")
    print(f"  Datos:         {total_bytes/1024:.1f} KB")
    print(f"{'─'*50}\n")

    # Abrir ambos puertos
    ser_tx = serial.Serial(port_tx, BAUD, timeout=2)
    ser_rx = serial.Serial(port_rx, BAUD, timeout=2)

    try:
        print("Esperando reset de las ESP32 (2s)...")
        time.sleep(2)
        ser_tx.reset_input_buffer()
        ser_rx.reset_input_buffer()

        # Lanzar hilo que lee logs del RX
        rx_stop = threading.Event()
        rx_thread = threading.Thread(target=rx_reader,
                                     args=(ser_rx, rx_stop), daemon=True)
        rx_thread.start()

        # ---- FASE 1: Enviar audio al TX ----
        print("\n[FASE 1] Enviando audio al ESP32 #1...")
        ser_tx.write(START_HEADER)
        ser_tx.write(struct.pack('<H', total_blocks))

        raw = audio[:total_blocks * N].tobytes()
        chunk_size = 1024
        sent = 0
        t0 = time.perf_counter()
        while sent < len(raw):
            chunk = raw[sent:sent + chunk_size]
            ser_tx.write(chunk)
            sent += len(chunk)
            pct = sent / len(raw) * 100
            bar = "#" * int(pct / 5)
            print(f"\r  [{bar:<20}] {pct:5.1f}%", end="", flush=True)

        ser_tx.write(END_FOOTER)
        ser_tx.flush()
        print(f"\n  Audio enviado en {time.perf_counter()-t0:.1f}s")

        # ---- FASE 2: Leer coeficientes del TX y reenviarlos al RX ----
        print(f"\n[FASE 2] Puenteando paquetes TX -> RX...")
        packets_ok = 0
        packets_fail = 0

        # Avisar al RX que empieza una sesion (cantidad de bloques)
        ser_rx.write(build_control_frame(1, total_blocks))
        ser_rx.flush()

        for i in range(total_blocks):
            pkt = read_packet(ser_tx, verbose)
            if pkt is None:
                packets_fail += 1
                print(f"\n  [{i+1}/{total_blocks}] ERROR: paquete no recibido")
                continue

            # Reenviar al receptor
            ser_rx.write(pkt)
            ser_rx.flush()
            packets_ok += 1

            pct = (i + 1) / total_blocks * 100
            bar = "#" * int(pct / 5)
            print(f"\r  [{bar:<20}] {pct:5.1f}%  {packets_ok} OK / {packets_fail} fail  (pkt {len(pkt)}B)",
                  end="", flush=True)

            # Cadencia real del audio: N/Fs = 256/8000 = 32ms por bloque
            # Esto permite al receptor reproducir sin gaps
            time.sleep(0.028)

        # Avisar al RX que terminó la sesion
        ser_rx.write(build_control_frame(2))
        ser_rx.flush()

        print(f"\n\n  Puenteados: {packets_ok}/{total_blocks}")
        print(f"  Fallidos:   {packets_fail}/{total_blocks}")

    except KeyboardInterrupt:
        print("\n\nInterrumpido.")
    finally:
        # Dar tiempo a que el RX termine de imprimir
        time.sleep(1)
        try: rx_stop.set()
        except: pass
        ser_tx.close()
        ser_rx.close()
        print("Puertos cerrados.")


# -----------------------------------------------------------
#  Entry point
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Envia .wav al ESP32 #1 y puentea coeficientes al ESP32 #2."
    )
    parser.add_argument("wav",       help="Archivo .wav")
    parser.add_argument("puerto_tx", help="Puerto del ESP32 #1 (Transmisor)")
    parser.add_argument("puerto_rx", help="Puerto del ESP32 #2 (Receptor)")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar logs del ESP32 #1")
    args = parser.parse_args()

    print(f"\nCargando '{args.wav}'...")
    try:
        audio = load_wav(args.wav)
    except FileNotFoundError:
        print(f"Error: no se encontro '{args.wav}'")
        sys.exit(1)

    print(f"  OK — {len(audio)} muestras @ {TARGET_FS} Hz ({len(audio)/TARGET_FS:.1f}s)")
    run_bridge(args.puerto_tx, args.puerto_rx, audio, verbose=args.verbose)


if __name__ == "__main__":
    main()
