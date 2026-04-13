import cmath
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sounddevice as sd
import os

# ============================================================
# FFT, IFFT y funciones espectrales
# ============================================================

def fft(x):
    """
    Transformada Rapida de Fourier — Cooley-Tukey radix-2 recursivo.
    Descompone la DFT de N puntos en dos DFTs de N/2:
        X[k] = FFT(par)[k]  +  W_N^k · FFT(impar)[k]
    donde W_N^k = e^{-j2πk/N} es el factor de giro.
    Complejidad O(N log N) frente a O(N²) de la DFT directa.
    """
    N = len(x)
    if N == 1:
        return [complex(x[0])]     # caso base: DFT de 1 punto = la muestra misma
    par   = fft(x[::2])            # DFT de muestras en posicion par  (0, 2, 4, ...)
    impar = fft(x[1::2])           # DFT de muestras en posicion impar (1, 3, 5, ...)
    W = [cmath.exp(-2j * cmath.pi * k / N) for k in range(N // 2)]  # factores de giro
    T = [W[k] * impar[k] for k in range(N // 2)]   # producto twiddle x impar
    # Combinacion mariposa: primera mitad suma, segunda resta
    return [par[k] + T[k] for k in range(N // 2)] + \
           [par[k] - T[k] for k in range(N // 2)]

def ifft(X):
    """
    IFFT usando la identidad conjugada:
        IFFT(X) = conj( FFT( conj(X) ) ) / N
    Reutiliza fft() sin duplicar codigo; el /N normaliza la energia.
    """
    N   = len(X)
    res = fft([v.conjugate() for v in X])   # FFT del espectro conjugado
    return [v.conjugate() / N for v in res]  # conjugar y escalar

def energia_bins(X, N):
    """
    Energia espectral por coeficiente — Teorema de Parseval discreto:
        E[k] = |X[k]|² / N
    La suma de todos los E[k] debe coincidir con la energia de x[n] en tiempo.
    """
    return [abs(X[k])**2 / N for k in range(N)]

def calcular_mse(x, x_rec):
    """
    Error Cuadratico Medio: MSE = (1/N) · Σ (x[n] - x̂[n])²
    Mide la distorsion promedio por muestra entre original y reconstruida.
    """
    N = len(x)
    return sum((float(x[n]) - x_rec[n].real)**2 for n in range(N)) / N

def bins_a_hz(N, fs):
    """
    Convierte indice de bin a frecuencia en Hz: fk = k · fs / N.
    Solo se devuelve la mitad positiva (N/2 bins); la negativa es simetrica.
    """
    return [k * fs / N for k in range(N // 2)]


# ============================================================
# 1. CAPTURA DE AUDIO
# ============================================================
# fs = 2^13 = 16384 Hz: potencia de 2 exacta, calidad superior a telefono
# (8000 Hz) y practicamente igual a FM (22050 Hz) para voz.
# Con DURACION=3 s se obtienen 3*16384 = 49152 muestras;
# la mayor potencia de 2 que cabe es 2^15 = 32768 (~2 s efectivos).
fs       = 2**13   # 16384 Hz
DURACION = 3

sep = "─" * 52

print("  COMPRESION ESPECTRAL — FFT/IFFT ")
print(sep)
print(f"  Grabando {DURACION} s a {fs} Hz.")
input("  Presionar ENTER para iniciar la grabacion")
print("  Captando senal")

grabacion = sd.rec(int(DURACION * fs), samplerate=fs, channels=1, dtype='float64')
sd.wait()   # bloquea hasta que termina la grabacion
print("  Grabacion completada.\n")

# sounddevice devuelve shape (N, 1); flatten() lo convierte en vector 1D
audio_raw = grabacion.flatten()

# Recortar a la mayor potencia de 2 <= muestras grabadas.
# Requerido por Cooley-Tukey: el algoritmo divide por 2 en cada nivel.
N = 2 ** int(np.floor(np.log2(len(audio_raw))))
x = audio_raw[:N]
t = np.arange(N) / fs   # eje de tiempo en segundos

print(f"  Muestras usadas (2^k): {N}  (2^{int(math.log2(N))})")

# PRE-PROCESAMIENTO:
# 1) Eliminar offset DC: evita que X[0] domine la energia y distorsione
#    la seleccion de componentes. El microfono introduce este sesgo.
# 2) Normalizar a [-1, 1]: permite comparar senales independientemente
#    del volumen de grabacion y evita amplitudes que desborden el DAC.
x = x - float(np.mean(x))
pico = float(np.max(np.abs(x)))
if pico > 0:
    x = x / pico


# ============================================================
# 2. FFT
# ============================================================
# Se convierte x (numpy array) a lista Python antes de llamar fft()
# porque nuestra implementacion trabaja con listas de complejos nativos.
print("\n  Calculando FFT")
X     = fft(x.tolist())      # lista de N numeros complejos X[0..N-1]
freqs = bins_a_hz(N, fs)     # eje de frecuencias [Hz] para la mitad positiva
E     = energia_bins(X, N)   # energia de cada coeficiente espectral
energia_total = sum(E)        # energia total (verificable con sum(x²)/N por Parseval)


# ============================================================
# 3. COMPRESION ESPECTRAL — umbral del 95%
# ============================================================
# Se conservan solo los coeficientes que acumulan el 95% de la energia.
# Los demas se ponen a cero: cuanto mas compresible es la senal,
# menos coeficientes se necesitan para llegar al umbral.
UMBRAL = 0.95

# Ordenar indices de mayor a menor energia (Python puro, sin numpy)
indices_orden = sorted(range(N), key=lambda i: -E[i])

# Acumular los coeficientes mas energeticos hasta superar el umbral
energia_acum      = 0.0
indices_retenidos = set()    # set para busqueda O(1) al construir X_comp
for idx in indices_orden:
    energia_acum += E[idx]
    indices_retenidos.add(idx)
    if energia_acum / energia_total >= UMBRAL:
        break

k                = len(indices_retenidos)
razon_compresion = (1 - k / N) * 100   # % de coeficientes eliminados

# Construir espectro comprimido: cero donde el coeficiente no fue seleccionado
X_comp = [X[i] if i in indices_retenidos else 0+0j for i in range(N)]


# ============================================================
# 4. IFFT — implementacion propia
# ============================================================
# La IFFT convierte el espectro comprimido de vuelta al dominio del tiempo.
# Se toma solo la parte real; la imaginaria residual es error numerico (~1e-15).
print("  Calculando IFFT")
x_rec    = ifft(X_comp)
x_rec_np = np.array([v.real for v in x_rec])   # convertir a array para graficas y audio


# ============================================================
# 5. METRICAS
# ============================================================
MSE                    = calcular_mse(x, x_rec)
energia_preservada_pct = (energia_acum / energia_total) * 100
potencia_senal         = sum(float(v)**2 for v in x) / N   # media de x[n]² (Python puro)
SNR_dB                 = 10 * math.log10(potencia_senal / MSE)  # dB: cuanto mas alto, mejor

# Energia acumulada ordenada para la grafica 7.4 (cumsum manual)
acum = 0.0
energia_ord_norm = []
for idx in indices_orden:
    acum += E[idx]
    energia_ord_norm.append(acum / energia_total * 100)

print(sep)
print(f"  Componentes totales N:     {N}")
print(f"  Componentes retenidas (k): {k}")
print(f"  Razon de compresion:       {razon_compresion:.4f}% coef. eliminados")
print(f"  Energia total (Parseval):  {energia_total:.6f}")
print(f"  Energia preservada:        {energia_preservada_pct:.4f}%")
print(sep)
print(f"  MSE:                       {MSE:.8f}")
print(f"  SNR (senal/error):         {SNR_dB:.2f} dB")
print(sep)


# ============================================================
# 6. REPRODUCCION DE AUDIO
# ============================================================
# Se vuelve a normalizar antes de reproducir porque la reconstruida
# puede tener un pico ligeramente distinto al original tras la compresion.
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 0 else s

print("\n  ── Reproduccion de audio ──────────────────")
input("  Presionar ENTER para escuchar la senal ORIGINAL")
sd.play(normalizar(x), samplerate=fs)
sd.wait()

input("  Presionar ENTER para escuchar la senal RECONSTRUIDA")
sd.play(normalizar(x_rec_np), samplerate=fs)
sd.wait()

print("  Reproduccion finalizada.")
print(sep)


# ============================================================
# 7. VISUALIZACION
# ============================================================
fig = plt.figure(figsize=(16, 13))
fig.suptitle(
    "Compresion Espectral Adaptativa — FFT/IFFT\n",
    fontsize=13, fontweight='bold', y=0.98
)
gs = GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

# Buscar la ventana de 1 s con mayor energia RMS.
# Los primeros instantes suelen ser silencio (antes de hablar),
# asi que mostrar el segmento mas activo da una comparacion mas honesta.
win_muestras = min(int(1.0 * fs), N)
hop          = max(win_muestras // 4, 1)
mejor_inicio, mejor_rms = 0, 0.0
for inicio in range(0, N - win_muestras + 1, hop):
    seg = x[inicio:inicio + win_muestras]
    rms = math.sqrt(sum(float(v)**2 for v in seg) / win_muestras)
    if rms > mejor_rms:
        mejor_rms, mejor_inicio = rms, inicio

ini, fin = mejor_inicio, mejor_inicio + win_muestras
t_seg    = t[ini:fin]

# 7.1 Senal original vs reconstruida en el dominio del tiempo.
# Se superponen para apreciar si la compresion introdujo artefactos visibles.
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_seg, x[ini:fin],        label='Original',
         color='steelblue', linewidth=1.2, alpha=0.9)
ax1.plot(t_seg, x_rec_np[ini:fin], label=f'Reconstruida (k={k})',
         color='tomato', linewidth=1.2, linestyle='--')
ax1.set_title(f"Senal Original vs Reconstruida  ({ini/fs:.2f}s – {fin/fs:.2f}s, mayor energia)")
ax1.set_xlabel("Tiempo [s]")
ax1.set_ylabel("Amplitud normalizada")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 7.2 y 7.3 Espectros de magnitud (mitad positiva).
# El factor 2/N compensa la simetria: la energia de la mitad negativa
# se "dobla" sobre la positiva para obtener amplitudes reales correctas.
N2       = N // 2
mag_orig = np.array([(2 / N) * abs(X[k])      for k in range(N2)])
mag_comp = np.array([(2 / N) * abs(X_comp[k]) for k in range(N2)])

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(freqs, mag_orig, color='steelblue', linewidth=0.7)
ax2.set_title("Espectro de Magnitud — Original")
ax2.set_xlabel("Frecuencia [Hz]")
ax2.set_ylabel("|X[k]| normalizado")
ax2.set_xlim([0, fs / 2])
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(freqs, mag_comp, color='tomato', linewidth=0.7)
ax3.set_title(f"Espectro Comprimido  ({k} de {N} coeficientes)")
ax3.set_xlabel("Frecuencia [Hz]")
ax3.set_ylabel("|X[k]| normalizado")
ax3.set_xlim([0, fs / 2])
ax3.grid(True, alpha=0.3)

# 7.4 Curva de energia acumulada en escala logaritmica.
# La mayoria de la energia se concentra en pocos coeficientes, por eso
# la curva sube bruscamente y luego se aplana. La linea verde indica
# exactamente cuantos coeficientes se necesitaron para llegar al 95%.
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(range(1, N + 1), energia_ord_norm, color='darkorange', linewidth=2)
ax4.axhline(y=95, color='red', linestyle='--', linewidth=1.3, label='Umbral 95%')
ax4.axvline(x=k, color='green', linestyle='--', linewidth=1.3, label=f'k = {k}')
ax4.fill_between(range(1, k + 1), energia_ord_norm[:k], alpha=0.15, color='green')
ax4.set_xscale('log')
ax4.set_xlabel("Numero de componentes (escala log)")
ax4.set_ylabel("Energia acumulada [%]")
ax4.set_title("Energia Acumulada vs N.o de Componentes")
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 102])

# 7.5 Error de reconstruccion: x[n] - x̂[n].
# Si el error es ruido blanco sin estructura, la compresion es uniforme.
# Si hay zonas con error alto, la compresion distorsiona esas regiones.
error = x[ini:fin] - x_rec_np[ini:fin]
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(t_seg, error, color='purple', linewidth=1.0)
ax5.axhline(y=0, color='black', linewidth=0.8)
ax5.set_title(
    f"Error de Reconstruccion  x[n] - x_rec[n]\n"
    f"MSE = {MSE:.6f}   |   SNR = {SNR_dB:.2f} dB   |   "
    f"Energia preservada = {energia_preservada_pct:.3f}%"
)
ax5.set_xlabel("Tiempo [s]")
ax5.set_ylabel("Error")
ax5.grid(True, alpha=0.3)

# Guardar en la misma carpeta del script para no depender del directorio
# desde donde se ejecute (os.path.abspath resuelve la ruta absoluta).
script_dir  = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "compresion_espectral.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nGrafica guardada en: {output_path}")