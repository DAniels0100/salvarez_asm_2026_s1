import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =============================================================================
# 1. GENERACION DE LA SENAL COMPUESTA (multicomponente)
# =============================================================================
fs = 1024           # Frecuencia de muestreo [Hz]
N  = 1024           # Numero de muestras — potencia de 2 (requerido por FFT)
t  = np.arange(N) / fs   # Vector de tiempo [s]

# Senal compuesta: suma de 5 tonos con distintas frecuencias y amplitudes
frecuencias = [50,  120,  200,  350,  500]    # [Hz]
amplitudes  = [3.0, 1.5,  2.0,  0.8,  0.5]   # Amplitudes
fases       = [0,   np.pi/4, np.pi/3, np.pi/6, np.pi/2]  # Fases [rad]

x = np.zeros(N)
for A, f, phi in zip(amplitudes, frecuencias, fases):
    x += A * np.sin(2 * np.pi * f * t + phi)

# Ruido gaussiano leve (simula condiciones reales de ADC)
np.random.seed(42)
x += 0.15 * np.random.randn(N)

# =============================================================================
# 2. CALCULO DE LA FFT
# =============================================================================
# La FFT calcula los coeficientes X[k] de la DFT:
#   X[k] = sum_{n=0}^{N-1} x[n] * e^{-j*2*pi*k*n/N}
#
# numpy usa el algoritmo Cooley-Tukey (divide y venceras), O(N log N)
X     = np.fft.fft(x)
freqs = np.fft.fftfreq(N, d=1.0/fs)   # Eje de frecuencias [Hz]

# Energia espectral por coeficiente (Teorema de Parseval discreto):
#   E_total = (1/N) * sum_k |X[k]|^2  =  sum_n |x[n]|^2
energia_espectral = (np.abs(X) ** 2) / N
energia_total     = np.sum(energia_espectral)

# =============================================================================
# 3. COMPRESION ESPECTRAL ADAPTATIVA — umbral del 95%
# =============================================================================
UMBRAL = 0.95   # Porcentaje minimo de energia a conservar

# Paso 1: ordenar todos los coeficientes de mayor a menor energia
indices_orden = np.argsort(energia_espectral)[::-1]   # descendente

# Paso 2: acumular energia hasta superar el umbral
energia_acum      = 0.0
indices_retenidos = []

for idx in indices_orden:
    energia_acum += energia_espectral[idx]
    indices_retenidos.append(idx)
    if energia_acum / energia_total >= UMBRAL:
        break

k = len(indices_retenidos)          # componentes necesarias
razon_compresion = (1 - k / N) * 100   # % de coeficientes eliminados

# Paso 3: crear espectro comprimido (ceros donde no se retuvo)
X_comp = np.zeros(N, dtype=complex)
X_comp[indices_retenidos] = X[indices_retenidos]

# =============================================================================
# 4. RECONSTRUCCION CON IFFT
# =============================================================================
# La IFFT calcula:
#   x_rec[n] = (1/N) * sum_k X_comp[k] * e^{j*2*pi*k*n/N}
#
# Al haber puesto a cero los coeficientes de menor energia, la senal
# reconstruida es una aproximacion de la original con menos informacion.
x_rec = np.fft.ifft(X_comp).real

# =============================================================================
# 5. METRICAS DE CALIDAD
# =============================================================================
# MSE (Error Cuadratico Medio)
MSE = np.mean((x - x_rec) ** 2)

# Energia preservada (%)
energia_preservada_pct = (energia_acum / energia_total) * 100

# SNR — Relacion senal a error [dB]
potencia_error = np.mean((x - x_rec) ** 2)
potencia_senal = np.mean(x ** 2)
SNR_dB = 10 * np.log10(potencia_senal / potencia_error)

# Energia acumulada normalizada para graficar la curva de compresion
energia_ord_norm = np.cumsum(energia_espectral[indices_orden]) / energia_total * 100

# =============================================================================
# REPORTE EN CONSOLA
# =============================================================================
sep = "─" * 50
print(sep)
print("  COMPRESION ESPECTRAL ADAPTATIVA — FFT/IFFT")
print(sep)
print(f"  Señal de {N} muestras a fs = {fs} Hz")
print(f"  Componentes totales N:          {N}")
print(f"  Componentes retenidas (k):      {k}")
print(f"  Razon de compresion:            {razon_compresion:.2f}% de coef. eliminados")
print(f"  Energia total (Parseval):       {energia_total:.4f}")
print(f"  Energia preservada:             {energia_preservada_pct:.4f}%")
print(sep)
print(f"  MSE:                            {MSE:.8f}")
print(f"  SNR (senal/error):              {SNR_dB:.2f} dB")
print(sep)

# =============================================================================
# 6. VISUALIZACION — 5 graficas
# =============================================================================
fig = plt.figure(figsize=(16, 13))
fig.suptitle(
    "Compresion Espectral Adaptativa — FFT / IFFT\n"
    "CE 1110 | TEC — Análisis de Señales Mixtas",
    fontsize=13, fontweight='bold', y=0.98
)
gs = GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

# ── 6.1  Señal original vs reconstruida (dominio tiempo) ──────────────────
ax1 = fig.add_subplot(gs[0, :])
muestra_fin = 300   # mostrar solo primeras 300 muestras para claridad
ax1.plot(t[:muestra_fin], x[:muestra_fin],
         label='Original', color='steelblue', linewidth=1.6, alpha=0.95)
ax1.plot(t[:muestra_fin], x_rec[:muestra_fin],
         label=f'Reconstruida  (k = {k} coeficientes)',
         color='tomato', linewidth=1.5, linestyle='--')
ax1.set_title("Señal Original vs Señal Reconstruida (primeras 300 muestras)")
ax1.set_xlabel("Tiempo [s]")
ax1.set_ylabel("Amplitud")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# ── 6.2  Espectro de magnitud — original (mitad positiva) ─────────────────
N2          = N // 2
freqs_pos   = freqs[:N2]
mag_orig    = (2 / N) * np.abs(X[:N2])
mag_comp_v  = (2 / N) * np.abs(X_comp[:N2])

ax2 = fig.add_subplot(gs[1, 0])
markerline, stemlines, baseline = ax2.stem(
    freqs_pos, mag_orig,
    linefmt='C0-', markerfmt='C0o', basefmt='k-')
plt.setp(stemlines, linewidth=0.8)
plt.setp(markerline, markersize=2)
ax2.set_title("Espectro de Magnitud — Original")
ax2.set_xlabel("Frecuencia [Hz]")
ax2.set_ylabel("|X[k]| normalizado")
ax2.set_xlim([0, fs / 2])
ax2.grid(True, alpha=0.3)

# ── 6.3  Espectro de magnitud — comprimido ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
markerline2, stemlines2, baseline2 = ax3.stem(
    freqs_pos, mag_comp_v,
    linefmt='C3-', markerfmt='C3o', basefmt='k-')
plt.setp(stemlines2, linewidth=0.8)
plt.setp(markerline2, markersize=2)
ax3.set_title(f"Espectro de Magnitud — Comprimido\n({k} de {N} coeficientes retenidos)")
ax3.set_xlabel("Frecuencia [Hz]")
ax3.set_ylabel("|X[k]| normalizado")
ax3.set_xlim([0, fs / 2])
ax3.grid(True, alpha=0.3)

# ── 6.4  Curva de energía acumulada vs nº de componentes ─────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(range(1, N + 1), energia_ord_norm, color='darkorange', linewidth=2)
ax4.axhline(y=95, color='red', linestyle='--', linewidth=1.3,
            label='Umbral 95%')
ax4.axvline(x=k, color='green', linestyle='--', linewidth=1.3,
            label=f'k = {k} componentes')
ax4.fill_between(range(1, k + 1), energia_ord_norm[:k],
                 alpha=0.15, color='green')
ax4.set_xscale('log')
ax4.set_xlabel("Número de componentes (escala log)")
ax4.set_ylabel("Energía acumulada [%]")
ax4.set_title("Energía Acumulada vs N.º de Componentes")
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 102])

# ── 6.5  Error de reconstrucción en el tiempo ─────────────────────────────
error = x - x_rec
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(t[:muestra_fin], error[:muestra_fin], color='purple', linewidth=1.1)
ax5.axhline(y=0, color='black', linewidth=0.8)
ax5.set_title(
    f"Error de Reconstrucción  x[n] − x̂[n]\n"
    f"MSE = {MSE:.6f}   |   SNR = {SNR_dB:.2f} dB   |   "
    f"Energía preservada = {energia_preservada_pct:.3f}%"
)
ax5.set_xlabel("Tiempo [s]")
ax5.set_ylabel("Error")
ax5.grid(True, alpha=0.3)

# ── Guardar ───────────────────────────────────────────────────────────────
# Guarda la imagen en la misma carpeta donde está este script
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "compresion_espectral.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nGráfica guardada en: {output_path}")