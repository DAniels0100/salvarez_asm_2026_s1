import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sounddevice as sd
import os


# ============================================================
# 1. CAPTURA DE AUDIO (3 segundos desde el micrófono)
# ============================================================
# Se trabaja a 44100 Hz porque es la frecuencia estándar de CD;
# cubre todo el rango audible humano hasta ~20 kHz sin problemas.
fs       = 44100
DURACION = 3

sep = "─" * 52
print(sep)
print("  COMPRESION ESPECTRAL ADAPTATIVA — FFT/IFFT")
print(sep)
print(f"  Se graba una señal de audio de {DURACION} segundos.")
input("  Presionar ENTER para iniciar la grabación")
print("  Captando señal")

grabacion = sd.rec(int(DURACION * fs), samplerate=fs, channels=1, dtype='float64')
sd.wait()
print("  Grabación completada.\n")

# Se aplana el array porque sounddevice devuelve forma (N, 1)
# y el resto del código espera un vector 1D.
audio_raw = grabacion.flatten()

# Se recorta a la potencia de 2 más cercana por debajo del total.
# Cooley-Tukey funciona mejor (y más rápido) con tamaños 2^k.
N = 2 ** int(np.floor(np.log2(len(audio_raw))))
x = audio_raw[:N]
t = np.arange(N) / fs

print(f"  Muestras grabadas:          {len(audio_raw)}")
print(f"  Muestras usadas (2^k):      {N}  (2^{int(np.log2(N))})")


# ── PRE-PROCESAMIENTO ─────────────────────────────────────────
# Se elimina el offset DC para que el coeficiente X[0] no
# "robe" energía al resto del espectro y distorsione el análisis.
# Luego se normaliza a [-1, 1] para comparar señales de
# distintas fuentes sin depender del volumen de grabación.
x = x - np.mean(x)
pico = np.max(np.abs(x))
if pico > 0:
    x = x / pico


# ============================================================
# 2. CÁLCULO DE LA FFT
# ============================================================
# Se aplica la FFT de NumPy, que usa Cooley-Tukey internamente.
# X[k] guarda cuánto hay de cada frecuencia en la señal.
X     = np.fft.fft(x)
freqs = np.fft.fftfreq(N, d=1.0 / fs)

# Se calcula la energía de cada coeficiente aplicando Parseval:
# la energía total en frecuencia debe coincidir con la del tiempo.
energia_espectral = (np.abs(X) ** 2) / N
energia_total     = np.sum(energia_espectral)


# ============================================================
# 3. COMPRESIÓN ESPECTRAL ADAPTATIVA — umbral del 95%
# ============================================================
# no todos los coeficientes valen lo mismo.
# Se conservan solo los que acumulan el 95% de la energía total
# y el resto se descarta. Cuanto más compresible es la señal,
# menos coeficientes se necesitan para llegar al umbral.
UMBRAL = 0.95

# Paso 1: se ordenan los coeficientes de mayor a menor energía
indices_orden = np.argsort(energia_espectral)[::-1]

# Paso 2: se van sumando los más energéticos hasta superar el umbral
energia_acum      = 0.0
indices_retenidos = []

for idx in indices_orden:
    energia_acum += energia_espectral[idx]
    indices_retenidos.append(idx)
    if energia_acum / energia_total >= UMBRAL:
        break

k                = len(indices_retenidos)
razon_compresion = (1 - k / N) * 100

# Paso 3: se construye el espectro comprimido poniendo a cero
# todos los coeficientes que no fueron seleccionados.
X_comp                    = np.zeros(N, dtype=complex)
X_comp[indices_retenidos] = X[indices_retenidos]


# ============================================================
# 4. RECONSTRUCCIÓN CON IFFT
# ============================================================
# Se aplica la IFFT al espectro comprimido para volver al dominio
# del tiempo. Se toma solo la parte real porque los errores
# numéricos dejan una parte imaginaria residual despreciable.
x_rec = np.fft.ifft(X_comp).real


# ============================================================
# 5. MÉTRICAS DE CALIDAD
# ============================================================
# MSE: error cuadrático medio entre original y reconstruida.
# SNR: relación señal/error en dB; cuanto más alto, mejor
# se preservó la señal. Con 95% de energía suele quedar alto.
MSE                    = np.mean((x - x_rec) ** 2)
energia_preservada_pct = (energia_acum / energia_total) * 100
SNR_dB                 = 10 * np.log10(np.mean(x ** 2) / MSE)
energia_ord_norm       = np.cumsum(energia_espectral[indices_orden]) / energia_total * 100

print(sep)
print(f"  Componentes totales N:          {N}")
print(f"  Componentes retenidas (k):      {k}")
print(f"  Razón de compresión:            {razon_compresion:.4f}% coef. eliminados")
print(f"  Energía total (Parseval):       {energia_total:.4f}")
print(f"  Energía preservada:             {energia_preservada_pct:.4f}%")
print(sep)
print(f"  MSE:                            {MSE:.8f}")
print(f"  SNR (señal/error):              {SNR_dB:.2f} dB")
print(sep)


# ============================================================
# 6. REPRODUCCIÓN DE AUDIO
# ============================================================
# Se normaliza de nuevo antes de reproducir por si la reconstruida
# tiene un pico ligeramente distinto al original tras la compresión.
def normalizar(senal):
    maximo = np.max(np.abs(senal))
    return senal / maximo if maximo > 0 else senal

x_norm     = normalizar(x)
x_rec_norm = normalizar(x_rec)

print("\n  ── Reproducción de audio ──────────────────")
input("  Presionar ENTER para escuchar la señal ORIGINAL")
sd.play(x_norm, samplerate=fs)
sd.wait()

input("  Presionar ENTER para escuchar la señal RECONSTRUIDA")
sd.play(x_rec_norm, samplerate=fs)
sd.wait()

print("  Reproducción finalizada.")
print(sep)


# ============================================================
# 7. VISUALIZACIÓN — gráficas
# ============================================================
fig = plt.figure(figsize=(16, 13))
fig.suptitle(
    "Compresión Espectral Adaptativa — FFT / IFFT  (señal de micrófono)\n",
    fontsize=13, fontweight='bold', y=0.98
)
gs = GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

# Se busca el segundo con mayor energía RMS para que la gráfica
# temporal muestre la parte más "activa" de la señal y no un
# fragmento de silencio que no dice nada.
win_muestras = int(1.0 * fs)
hop          = win_muestras // 4
mejor_inicio = 0
mejor_rms    = 0.0
for inicio in range(0, N - win_muestras, hop):
    rms = np.sqrt(np.mean(x[inicio : inicio + win_muestras] ** 2))
    if rms > mejor_rms:
        mejor_rms    = rms
        mejor_inicio = inicio

muestra_ini = mejor_inicio
muestra_fin = mejor_inicio + win_muestras
t_seg       = t[muestra_ini:muestra_fin]
print(f"  Segmento mostrado:  {muestra_ini/fs:.3f} s – {muestra_fin/fs:.3f} s  (mayor energía RMS)")

# ── 7.1  Señal original vs reconstruida (dominio tiempo) ─────
# Se superponen ambas señales para apreciar a ojo si la compresión
# introdujo artefactos visibles. Con buen SNR deben ser casi indistinguibles.
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_seg, x[muestra_ini:muestra_fin],
         label='Original', color='steelblue', linewidth=1.2, alpha=0.9)
ax1.plot(t_seg, x_rec[muestra_ini:muestra_fin],
         label=f'Reconstruida  (k = {k} coeficientes)',
         color='tomato', linewidth=1.2, linestyle='--')
ax1.set_title(f"Señal de Audio Original vs Reconstruida  "
              f"(1 s de mayor energía: {muestra_ini/fs:.2f}s – {muestra_fin/fs:.2f}s)")
ax1.set_xlabel("Tiempo [s]")
ax1.set_ylabel("Amplitud normalizada")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# ── 7.2  Espectro de magnitud — original ─────────────────────
# Solo se grafica la mitad positiva; la negativa es simétrica
# para señales reales y no aporta información adicional.
N2        = N // 2
freqs_pos = freqs[:N2]
mag_orig  = (2 / N) * np.abs(X[:N2])
mag_comp  = (2 / N) * np.abs(X_comp[:N2])

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(freqs_pos, mag_orig, color='steelblue', linewidth=0.6)
ax2.set_title("Espectro de Magnitud — Original")
ax2.set_xlabel("Frecuencia [Hz]")
ax2.set_ylabel("|X[k]| normalizado")
ax2.set_xlim([0, fs / 2])
ax2.grid(True, alpha=0.3)

# ── 7.3  Espectro de magnitud — comprimido ───────────────────
# Se espera ver los mismos picos que en el original pero con
# el piso espectral silenciado; los "huecos" son los coeficientes eliminados.
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(freqs_pos, mag_comp, color='tomato', linewidth=0.6)
ax3.set_title(f"Espectro de Magnitud — Comprimido\n({k} de {N} coeficientes retenidos)")
ax3.set_xlabel("Frecuencia [Hz]")
ax3.set_ylabel("|X[k]| normalizado")
ax3.set_xlim([0, fs / 2])
ax3.grid(True, alpha=0.3)

# ── 7.4  Curva de energía acumulada vs nº de componentes ─────
# Se muestra en escala logarítmica porque la mayor parte de la energía
# se concentra en pocos coeficientes; la curva sube rápido al inicio
# y luego se aplana. La línea verde marca exactamente dónde se cortó.
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(range(1, N + 1), energia_ord_norm, color='darkorange', linewidth=2)
ax4.axhline(y=95, color='red', linestyle='--', linewidth=1.3, label='Umbral 95%')
ax4.axvline(x=k, color='green', linestyle='--', linewidth=1.3,
            label=f'k = {k} componentes')
ax4.fill_between(range(1, k + 1), energia_ord_norm[:k], alpha=0.15, color='green')
ax4.set_xscale('log')
ax4.set_xlabel("Número de componentes (escala log)")
ax4.set_ylabel("Energía acumulada [%]")
ax4.set_title("Energía Acumulada vs N.º de Componentes")
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 102])

# ── 7.5  Error de reconstrucción en el tiempo ─────────────────
# Se grafica x[n] - x̂[n] para ver si el error es ruido blanco
# (bien distribuido, sin estructura) o si hay zonas donde la
# compresión introduce distorsión sistemática.
error = x - x_rec
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(t_seg, error[muestra_ini:muestra_fin], color='purple', linewidth=1.0)
ax5.axhline(y=0, color='black', linewidth=0.8)
ax5.set_title(
    f"Error de Reconstrucción  x[n] − x̂[n]\n"
    f"MSE = {MSE:.6f}   |   SNR = {SNR_dB:.2f} dB   |   "
    f"Energía preservada = {energia_preservada_pct:.3f}%"
)
ax5.set_xlabel("Tiempo [s]")
ax5.set_ylabel("Error")
ax5.grid(True, alpha=0.3)

# Se guarda en la misma carpeta del script para no depender
# del directorio de trabajo actual desde donde se ejecute.
script_dir  = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "compresion_espectral.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nGráfica guardada en: {output_path}")