% =========================================================
%  CE-1110 — Analisis de Senales Mixtas
%  Proyecto Grupal 2: Sistema de control de ascensor
%  Caracterizacion del sistema — Punto 7 de Tarea 1
%
%  NO requiere pkg control ni ningun paquete externo.
%
%  Modelo de la planta (2do orden con integrador):
%       G(s) = Ktotal / [ s*(tau*s + 1) ]
%
%  Respuesta al escalon (ec. 16 del modelo matematico):
%       theta(t) = Ktotal*A * [ t - tau*(1 - e^(-t/tau)) ]
%
%  Respuesta al impulso (ec. 17):
%       theta(t) = Ktotal * (1 - e^(-t/tau))
% =========================================================

clear; clc; close all;

% =========================================================
%  SECCION 1: Parametros del sistema
%  Reemplaza con tus valores medidos experimentalmente.
% =========================================================

Ra   = 5.0;    % [Ohm]        Resistencia de armadura (medir con multimetro)
Kt   = 0.02;   % [N*m/A]     Constante de torque (estimado, motor hobby sin hoja de datos)
Kb   = 0.02;   % [V*s/rad]   Constante de back-EMF (= Kt en motor DC ideal)
J    = 0.0005; % [kg*m^2]    Momento de inercia (cabina de papel + polea pequena, sin carga)
B    = 0.0001; % [N*m*s/rad] Friccion viscosa (despreciable, cabina de papel)

KPWM = 1.0;
KPH  = 0.95;
Kpot = 0.5;    % [V/rad] calibrar con el potenciometro real

% Parametros de simulacion
t_final = 3.0;   % [s] 60 cm se recorre rapido con motor de hobby
dt      = 0.001;
A_step  = 6.0;   % [V] amplitud del escalon (voltaje de alimentacion del L298N)

% =========================================================
%  SECCION 2: Calculo del modelo
%  Ecuaciones (10)-(14) del documento matematico
% =========================================================

denom_motor = Ra*B + Kt*Kb;
K_motor     = Kt / denom_motor;       % Ganancia DC del motor
tau         = Ra*J / denom_motor;     % Constante de tiempo [s]
Ktotal      = KPWM * KPH * K_motor * Kpot;

fprintf('=== Parametros del modelo ===\n');
fprintf('  K_motor = %.4f\n', K_motor);
fprintf('  tau     = %.4f s\n', tau);
fprintf('  Ktotal  = %.4f\n', Ktotal);
fprintf('  Polo p1 = 0  (integrador)\n');
fprintf('  Polo p2 = %.4f rad/s\n', -1/tau);

% =========================================================
%  SECCION 3: Senales en el tiempo (formulas analiticas)
% =========================================================

t = 0:dt:t_final;

% Respuesta al impulso: theta(t) = Ktotal*(1 - e^(-t/tau))
theta_impulso = Ktotal .* (1 - exp(-t ./ tau));

% Respuesta al escalon: theta(t) = Ktotal*A*[t - tau*(1-e^(-t/tau))]
theta_escalon = Ktotal .* A_step .* (t - tau .* (1 - exp(-t ./ tau)));

% Rampa pura (sin transitorio) — referencia
rampa_pura = Ktotal .* A_step .* t;

% =========================================================
%  SECCION 4: Identificacion de tau y K desde la curva
%  Procedimiento de la Seccion 1.8 del modelo matematico
% =========================================================

valor_ss   = Ktotal;
umbral_63  = 0.632 * valor_ss;
idx_tau    = find(theta_impulso >= umbral_63, 1, 'first');
tau_medido = t(idx_tau);

fprintf('\n=== Identificacion desde la curva de impulso ===\n');
fprintf('  Valor estado estacionario : %.4f\n', valor_ss);
fprintf('  tau al 63.2%%              : %.4f s  (teorico: %.4f s)\n', ...
        tau_medido, tau);

% =========================================================
%  SECCION 5: Polos del sistema
%  G(s) = Ktotal / [tau*s^2 + s]
%  Raices: s*(tau*s+1)=0  =>  p1=0,  p2=-1/tau
% =========================================================

p1 = 0;
p2 = -1 / tau;

fprintf('\n=== Polos de G(s) ===\n');
fprintf('  p1 = 0          (integrador puro)\n');
fprintf('  p2 = %.4f rad/s\n', p2);
fprintf('  Sin ceros finitos\n');

% =========================================================
%  SECCION 6: Frecuencia de muestreo sugerida
%  Criterio: fs >= 10 x ancho de banda del sistema
%  BW aprox = |p2| / (2*pi)  [Hz]
% =========================================================

BW_hz       = abs(p2) / (2*pi);
fs_sugerida = max(10 * BW_hz * 5, 100);   % minimo 100 Hz
Ts_sugerido = 1 / fs_sugerida;

fprintf('\n=== Frecuencia de muestreo ===\n');
fprintf('  Ancho de banda aprox. : %.4f Hz\n', BW_hz);
fprintf('  fs sugerida           : %.1f Hz\n', fs_sugerida);
fprintf('  Ts sugerido           : %.5f s\n', Ts_sugerido);

% =========================================================
%  SECCION 7: Figuras
% =========================================================

figure('Name', 'CE-1110 — Planta del ascensor', ...
       'NumberTitle', 'off', 'Position', [50, 50, 1100, 750]);

% ---- (a) Respuesta al impulso ----
subplot(2, 2, 1);
plot(t, theta_impulso, 'b-', 'LineWidth', 2);
hold on;
yline(Ktotal,    'r--', 'LineWidth', 1.2);
xline(tau_medido,'g-.', 'LineWidth', 1.2);
plot(tau_medido, umbral_63, 'go', 'MarkerSize', 9, 'MarkerFaceColor', 'g');
xlabel('Tiempo [s]');
ylabel('theta(t) [rad]');
title('Respuesta al impulso');
legend('theta(t)', ...
       ['Ktotal = ' num2str(Ktotal, '%.3f')], ...
       ['tau = '    num2str(tau_medido, '%.3f') ' s'], ...
       'Punto 63.2%', 'Location', 'southeast');
grid on;

% ---- (b) Respuesta al escalon ----
subplot(2, 2, 2);
plot(t, theta_escalon, 'b-', 'LineWidth', 2);
hold on;
plot(t, rampa_pura, 'r--', 'LineWidth', 1.2);
xlabel('Tiempo [s]');
ylabel('theta(t) [rad]');
title(['Respuesta al escalon  (V = ' num2str(A_step) ' V)']);
legend('theta(t) real', 'Rampa ideal Ktotal*A*t', 'Location', 'northwest');
grid on;
text(0.3, max(theta_escalon)*0.5, ...
     {'Sin controlador:', 'el ascensor no se detiene'}, ...
     'FontSize', 9, 'Color', 'r');

% ---- (c) Diagrama de polos y ceros (manual, sin paquete) ----
subplot(2, 2, 3);
hold on;

margen = abs(p2) * 0.5;
xlim([p2 - margen,  abs(p2)*0.3]);
ylim([-abs(p2)*0.4, abs(p2)*0.4]);

xline(0, 'k-', 'LineWidth', 0.8);
yline(0, 'k-', 'LineWidth', 0.8);

plot(p1, 0, 'rx', 'MarkerSize', 16, 'LineWidth', 2.5);
plot(p2, 0, 'rx', 'MarkerSize', 16, 'LineWidth', 2.5);

text(p1 + margen*0.05,  abs(p2)*0.10, ...
     'p1 = 0  (integrador)', 'FontSize', 9, 'Color', 'r');
text(p2 - margen*0.02, -abs(p2)*0.14, ...
     ['p2 = ' num2str(p2, '%.2f') ' rad/s'], ...
     'FontSize', 9, 'Color', 'r', 'HorizontalAlignment', 'right');

xlabel('Re(s)  [rad/s]');
ylabel('Im(s)  [rad/s]');
title('Diagrama de polos y ceros — G(s)');
legend('Polo (x)', 'Location', 'northeast');
grid on;

% ---- (d) Transitorio vs rampa ideal ----
subplot(2, 2, 4);
diferencia = rampa_pura - theta_escalon;
plot(t, theta_escalon, 'b-', 'LineWidth', 2);
hold on;
plot(t, rampa_pura,  'r--', 'LineWidth', 1.2);
plot(t, diferencia,  'g-',  'LineWidth', 1.5);
xlabel('Tiempo [s]');
ylabel('theta(t) [rad]');
title('Efecto del transitorio exponencial');
legend('theta(t) real', 'Rampa ideal', ...
       'Error = Ktotal*A*tau*(1-e^{-t/tau})', ...
       'Location', 'northwest');
grid on;

sgtitle('CE-1110 — Caracterizacion de la planta (sin controlador)', ...
        'FontSize', 12, 'FontWeight', 'bold');

% =========================================================
%  SECCION 8: Tabla resumen en consola
% =========================================================

fprintf('\n=== Tabla resumen de parametros ===\n');
fprintf('%-22s %-10s %-20s %s\n', 'Parametro', 'Simbolo', 'Valor', 'Metodo');
fprintf('%s\n', repmat('-', 1, 74));
fprintf('%-22s %-10s %-20s %s\n', 'Resist. armadura',    'Ra',   [num2str(Ra) ' Ohm'],              'Multimetro');
fprintf('%-22s %-10s %-20s %s\n', 'Constante de tiempo', 'tau',  [num2str(tau,'%.4f') ' s'],         'Curva escalon 63.2%');
fprintf('%-22s %-10s %-20s %s\n', 'Ganancia motor',      'K',    num2str(K_motor,'%.4f'),            'Respuesta al impulso');
fprintf('%-22s %-10s %-20s %s\n', 'Ganancia total',      'Ktot', num2str(Ktotal,'%.4f'),             'Cadena directa');
fprintf('%-22s %-10s %-20s %s\n', 'Polo dominante',      'p2',   [num2str(p2,'%.2f') ' rad/s'],      'Modelo');
fprintf('%-22s %-10s %-20s %s\n', 'Ganancia potenc.',    'Kpot', [num2str(Kpot) ' V/rad'],          'Calibracion');
fprintf('%-22s %-10s %-20s %s\n', 'Frec. muestreo',      'fs',   [num2str(fs_sugerida,'%.0f') ' Hz'],'Criterio 10x BW');

fprintf('\nListo.\n');