import numpy as np
import matplotlib.pyplot as plt
import time


# ============================================================
# 1. SEÑAL DE PRUEBA
# ============================================================
# Tse trabaja a 8 kHz porque es el mínimo estándar para voz
# y nos deja ver bien los tres componentes que vamos a meter.
fs = 8000

# se genera un bloque largo para poder recortar pedazos
# más adelante sin tener que regenerar nada.
N_base = 2048
t_base = np.arange(N_base) / fs

# Tres sinusoides sumadas: una fundamental grave (440 Hz,
# un La musical), una media (1200 Hz) y una aguda (2000 Hz).
# Las amplitudes van bajando para que el espectro no sea
# totalmente plano.
f1, f2, f3 = 440, 1200, 2000
x_base = (
    1.0 * np.sin(2 * np.pi * f1 * t_base) +
    0.6 * np.sin(2 * np.pi * f2 * t_base + np.pi / 4) +
    0.3 * np.sin(2 * np.pi * f3 * t_base + np.pi / 2)
)


# ============================================================
# 2. DFT IMPLEMENTADA DIRECTAMENTE
# ============================================================
def dft(signal: np.ndarray) -> np.ndarray:
    """
    DFT de manual: dos bucles anidados, O(N²).
    Lento, sí, pero deja ver exactamente lo que hace
    la fórmula sin ninguna magia por debajo.
    """
    N = len(signal)
    X = np.zeros(N, dtype=complex)

    # Para cada "cajón" de frecuencia k...
    for k in range(N):
        suma = 0j
        # se multiplica cada muestra por su exponencial compleja
        # y se acumula. Esto es la definición de la DFT.
        for n in range(N):
            suma += signal[n] * np.exp(-2j * np.pi * k * n / N)
        X[k] = suma

    return X


# ============================================================
# 3. FFT USANDO NUMPY
# ============================================================
def fft_numpy(signal: np.ndarray) -> np.ndarray:
    """
    El mismo resultado que dft(), pero NumPy usa el algoritmo
    Cooley-Tukey por debajo: divide el problema a la mitad
    en cada paso y lo baja de O(N²) a O(N log N).
    Para N=512 la diferencia empieza a notarse mucho.
    """
    return np.fft.fft(signal)


# ============================================================
# 4. FIGURAS DE ANÁLISIS ESPECTRAL
# ============================================================
# 512 muestras es el punto dulce: la DFT manual tarda unos
# pocos segundos (aguantable) y el espectro tiene suficiente
# resolución para distinguir bien las tres frecuencias.
N_fig = 512
x_fig = x_base[:N_fig]
t_fig = np.arange(N_fig) / fs

# Calculamos ambas versiones para comparar después
X_dft = dft(x_fig)
X_fft = fft_numpy(x_fig)

# fftfreq devuelve las frecuencias en orden "natural" de la FFT:
# primero 0..fs/2, luego las frecuencias negativas.
freqs = np.fft.fftfreq(N_fig, d=1 / fs)

# Solo nos interesa la mitad positiva; la negativa es simétrica
# para señales reales y no aporta información nueva.
N2 = N_fig // 2
freqs_pos = freqs[:N2]

# Factor 2/N: el 1/N normaliza la magnitud y el 2 compensa
# la energía que descartamos al quedarnos con un solo lado.
mag_dft = (2 / N_fig) * np.abs(X_dft[:N2])
mag_fft = (2 / N_fig) * np.abs(X_fft[:N2])

fase_fft = np.angle(X_fft[:N2])


# ------------------------------------------------------------
# Figura 1: señal en tiempo + espectro DFT
# Sirve para ver que la señal "parece compleja" en el tiempo
# pero el espectro la desmonta en sus tres piezas simples.
# ------------------------------------------------------------
fig1, ax = plt.subplots(2, 1, figsize=(10, 7))

ax[0].plot(t_fig, x_fig, linewidth=1.0)
ax[0].set_title("Señal de prueba en el dominio del tiempo")
ax[0].set_xlabel("Tiempo [s]")
ax[0].set_ylabel("Amplitud")
ax[0].grid(True, alpha=0.3)

ax[1].plot(freqs_pos, mag_dft, linewidth=1.0)
ax[1].set_title("Espectro de magnitud obtenido mediante la DFT")
ax[1].set_xlabel("Frecuencia [Hz]")
ax[1].set_ylabel("Magnitud")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("dft_resultado.png", dpi=150, bbox_inches="tight")


# ------------------------------------------------------------
# Figura 2: espectro FFT por separado para comparar visualmente
# con la figura anterior. el resultado esperado es que las dos gráficas
# de espectro se vean idénticas, confirmando que la FFT es correcta.
# ------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(10, 4))

ax.plot(freqs_pos, mag_fft, linewidth=1.0)
ax.set_title("Espectro de magnitud obtenido mediante la FFT")
ax.set_xlabel("Frecuencia [Hz]")
ax.set_ylabel("Magnitud")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fft_resultado.png", dpi=150, bbox_inches="tight")


# ------------------------------------------------------------
# Figura 3: magnitud y fase juntas
# La fase suele ser ruidosa en todo el espectro, pero cerca
# de 440, 1200 y 2000 Hz debería reflejar los desfases
# que le metimos a la señal (0, π/4 y π/2).
# ------------------------------------------------------------
fig3, ax = plt.subplots(2, 1, figsize=(10, 7))

ax[0].plot(freqs_pos, mag_fft, linewidth=1.0)
ax[0].set_title("Magnitud del espectro de la señal")
ax[0].set_xlabel("Frecuencia [Hz]")
ax[0].set_ylabel("Magnitud")
ax[0].grid(True, alpha=0.3)

ax[1].plot(freqs_pos, fase_fft, linewidth=1.0)
ax[1].set_title("Fase del espectro de la señal")
ax[1].set_xlabel("Frecuencia [Hz]")
ax[1].set_ylabel("Fase [rad]")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("magnitud_fase.png", dpi=150, bbox_inches="tight")


# ============================================================
# 5. COMPARACIÓN DE TIEMPOS PARA DISTINTOS TAMAÑOS N
# ============================================================
# aquí queremos ver cómo explota el tiempo de la DFT cuando N crece, mientras
# que la FFT prácticamente no se inmuta.

N_values = [64, 128, 256, 512, 1024]
tiempos_dft = []
tiempos_fft = []

print("Comparación de tiempos de ejecución")
print("-" * 60)
print(f"{'N':>6} | {'DFT [s]':>12} | {'FFT [s]':>12} | {'DFT/FFT':>10}")
print("-" * 60)

for N in N_values:
    xN = x_base[:N]

    # perf_counter es más preciso que time() para medir
    # intervalos cortos dentro del mismo proceso.
    inicio_dft = time.perf_counter()
    _ = dft(xN)
    fin_dft = time.perf_counter()

    inicio_fft = time.perf_counter()
    _ = fft_numpy(xN)
    fin_fft = time.perf_counter()

    tdft = fin_dft - inicio_dft
    tfft = fin_fft - inicio_fft

    # cuántas veces más rápida es la FFT. Con N=1024 suele superar las 100x.
    razon = tdft / tfft if tfft > 0 else np.inf

    tiempos_dft.append(tdft)
    tiempos_fft.append(tfft)

    print(f"{N:6d} | {tdft:12.6f} | {tfft:12.6f} | {razon:10.2f}")

print("-" * 60)


# ------------------------------------------------------------
# Figura 4: la aceleración de la FFT se ve de golpe en la
# gráfica: la línea de la DFT sube en curva cuadrática
# mientras la FFT se queda casi plana en el eje.
# ------------------------------------------------------------
fig4, ax = plt.subplots(figsize=(8, 5))

ax.plot(N_values, tiempos_dft, marker='o', label='DFT')
ax.plot(N_values, tiempos_fft, marker='o', label='FFT')
ax.set_title("Comparación de tiempos de ejecución entre DFT y FFT")
ax.set_xlabel("Número de muestras N")
ax.set_ylabel("Tiempo [s]")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("tiempos_dft_fft.png", dpi=150, bbox_inches="tight")


# ============================================================
# 6. MOSTRAR TODAS LAS FIGURAS
# ============================================================

plt.show()