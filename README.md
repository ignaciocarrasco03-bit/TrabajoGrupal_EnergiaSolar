import math  # Importa la biblioteca math para operaciones matemáticas como potencias.

eficiencia_ciclo = 0.4693  # Define la eficiencia de primera ley del ciclo Brayton, obtenida de la literatura (Wang et al., 2019).

potencia_neta = 100  # MW - Establece la potencia neta de generación nominal en 100 MW.

calor_PC = potencia_neta / eficiencia_ciclo  # MW - Calcula el calor requerido en el ciclo de potencia dividiendo la potencia neta por la eficiencia.

print(f'Calor_PC: {calor_PC:.3f} MW')  # Imprime el valor del calor requerido en el ciclo de potencia con 3 decimales.

a = 1443  # Coeficiente constante 'a' para el cálculo del calor específico de las sales fundidas.

b = 0.172  # Coeficiente constante 'b' para el cálculo del calor específico de las sales fundidas.

T_fria = 239  # Temperatura fría de entrada al campo solar en °C (239°C).

T_caliente = 600  # Temperatura caliente de salida del campo solar en °C (600°C).

DeltaT = T_caliente - T_fria  # Calcula la diferencia de temperaturas entre caliente y fría.

diff_cuadrada = T_caliente**2 - T_fria**2  # Calcula la diferencia de los cuadrados de las temperaturas para la integral del calor específico.

Delta_H = a * DeltaT + (b / 2) * diff_cuadrada  # J/kg - Calcula el cambio de entalpía integrando el calor específico sobre el rango de temperaturas.

print(f'Delta_H: {Delta_H:.2f} J/kg')  # Imprime el cambio de entalpía con 2 decimales.

flujo_masico_total = (calor_PC * 1e6) / Delta_H  # kg/s - Calcula el flujo másico total de sales convirtiendo MW a W y dividiendo por el cambio de entalpía.

print(f'flujo_masico_total: {flujo_masico_total:.2f} kg/s')  # Imprime el flujo másico total con 2 decimales.

# Campo solar - Sección para el diseño del campo solar.

area_loop = 574.77  # m2 - Área activa por loop (fila) de colectores.

area_modulo = 47.89  # m2 por módulo - Área activa por módulo individual.

num_modulos = 12  # Número de módulos por loop.

IAM = 0.9909  # Índice modificador por ángulo de incidencia, fijo en 0.9909.

eta0 = 0.84  # Eficiencia óptica máxima.

eta_suciedad = 0.97  # Factor de suciedad del espejo.

optica = IAM * eta0 * eta_suciedad  # Calcula la eficiencia óptica total multiplicando los factores ópticos.

print(f'optica: {optica:.4f}')  # Imprime la eficiencia óptica con 4 decimales.

DNI = 1040  # W/m2 - Irradiancia normal directa de diseño.

T_ambiente = 19.5  # Temperatura ambiental en °C.

def simular_T_final(flujo_loop):  # Define una función para simular la temperatura final al final de un loop dado un flujo másico por loop.
    T = T_fria  # Inicializa la temperatura actual con la temperatura fría de entrada.
    for _ in range(num_modulos):  # Itera sobre cada módulo en el loop (12 veces).
        delta_temp = T - T_ambiente  # Calcula la diferencia de temperatura entre la actual y la ambiental.
        perdida1 = 1.5 * delta_temp / DNI  # Calcula la primera pérdida térmica lineal (corregida de 1.5 a 0.15 para viabilidad).
        perdida2 = 0.0016 * delta_temp**2 / DNI  # Calcula la segunda pérdida térmica cuadrática.
        eficiencia = optica - perdida1 - perdida2  # Calcula la eficiencia de la fila restando las pérdidas de la óptica.
        if eficiencia < 0:  # Verifica si la eficiencia es negativa.
            eficiencia = 0  # Establece la eficiencia en 0 si es negativa para evitar valores no físicos.
        dQ = eficiencia * DNI * area_modulo  # W - Calcula el calor absorbido por módulo multiplicando eficiencia, DNI y área.
        Cp = 1443 + 0.172 * T  # J/kg°C - Calcula el calor específico a la temperatura actual.
        dTemp = dQ / (flujo_loop * Cp) if flujo_loop > 0 else 0  # Calcula el incremento de temperatura dividiendo calor por (flujo * Cp), o 0 si flujo es 0.
        T += dTemp  # Actualiza la temperatura sumando el incremento.
    return T  # Retorna la temperatura final al final del loop.

# Búsqueda binaria para encontrar el número mínimo de loops (N).

bajo = 1  # Inicializa el límite inferior de la búsqueda binaria en 1.

alto = 1000000  # Inicializa el límite superior de la búsqueda binaria en 10000 (un valor alto arbitrario).

while bajo < alto:  # Continúa el bucle mientras el límite inferior sea menor que el superior.
    medio = (bajo + alto) // 2  # Calcula el punto medio entero entre bajo y alto.
    flujo_l = flujo_masico_total / medio  # Calcula el flujo másico por loop dividiendo el total por el número medio de loops.
    T_final = simular_T_final(flujo_l)  # Simula la temperatura final para ese número de loops.
    if T_final >= T_caliente:  # Verifica si la temperatura final alcanza o supera la caliente requerida.
        alto = medio  # Si sí, reduce el límite superior a medio (busca un N menor).
    else:  # Si no.
        bajo = medio + 1  # Aumenta el límite inferior a medio + 1 (necesita más loops).

N = bajo  # Asigna el número mínimo de loops encontrado al final de la búsqueda.

print(f'Número de loops: {N}')  # Imprime el número de loops.

area = N * area_loop  # Calcula el área total multiplicando N por el área por loop.

print(f'Área asociada: {area:.2f} m2')  # Imprime el área asociada con 2 decimales.

T_final = simular_T_final(flujo_masico_total / N)  # Simula la temperatura final con el N encontrado para verificación.

print(f'T_final para N: {T_final:.2f} °C')  # Imprime la temperatura final verificada con 2 decimales.

#==================
#Pregunta 3.2.2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ruta_csv = r"C:\Users\HP\Downloads\DHTMY_E_S2CGKV.csv"  # ← AJUSTAR RUTA LOCAL
# Se carga el TMY y se omiten las primeras 41 filas del encabezado del archivo.
df = pd.read_csv(ruta_csv, skiprows=41)

# DNI horario real del sitio
DNI_horario = df['dni'].values.astype(float)

# Temperatura ambiente horaria (opcional, mejora precisión)
T_ambiente_horaria = df['temp'].values.astype(float)

print(f"DNI cargado: {len(DNI_horario)} valores, promedio {DNI_horario.mean():.2f} W/m²")

flujo_por_loop = flujo_masico_total / N

# Cálculo de eficiencia instantánea del colector solar parabólico
def eficiencia_colector(T, DNI_local, Tamb):
    delta_T = T - Tamb  # Diferencia entre fluido y ambiente

    # Pérdidas térmicas: lineal y cuadrática
    perdida1 = 1.5 * delta_T / DNI_local      # pérdida lineal
    perdida2 = 0.0016 * delta_T**2 / DNI_local # pérdida cuadrática

    # Eficiencia óptica total ya calculada antes: optica = IAM * eta0 * eta_suciedad
    eta = optica - perdida1 - perdida2

    return max(eta, 0)  # evita valores no físicos (no puede ser negativa)


# Cálculo del calor entregado por el campo solar cada hora (MWt)
def calcular_Q_campo_horario(DNI_vec, Tamb_vec):
    Q_campo = np.zeros(len(DNI_vec))
    for h in range(len(DNI_vec)):
        DNI_h = DNI_vec[h]
        Tamb = Tamb_vec[h]

        if DNI_h < 10:  # DNI muy bajo → no hay producción solar útil
            continue

        T = T_fria  # reinicia temperatura de entrada del fluido
        Q_loop_h = 0.0

        for _ in range(num_modulos):
            eta = eficiencia_colector(T, DNI_h, Tamb)
            dQ = eta * DNI_h * area_modulo  # W por módulo
            Q_loop_h += dQ
            Cp = a + b * T
            dT = dQ / (flujo_por_loop * Cp)
            T += dT

        Q_campo[h] = (Q_loop_h * N) / 1e6  # convierte a MW térmicos
    return Q_campo

# === CÁLCULO DEL CAMPO SOLAR HORARIO ===
Q_campo_horario = calcular_Q_campo_horario(DNI_horario, T_ambiente_horaria)

# Límite de potencia del ciclo Brayton (máx. 100 MW eléctricos)
W_neto_horario = eficiencia_ciclo * np.minimum(Q_campo_horario, calor_PC)  # MW

# Gráfico de potencia neta anual
plt.figure(figsize=(12, 6))
plt.plot(W_neto_horario, color='blue')
plt.title('3.2.2 Potencia neta por hora (SM = 1, sin TES)')
plt.xlabel('Hora del año')
plt.ylabel('Potencia neta (MW)')
plt.grid(True)
plt.savefig('potencia_horaria_322.png')
plt.show()

# Cálculo extra para informe (no cambiar)
produccion_anual = np.sum(W_neto_horario) / 1000  # GWh
factor_capacidad = np.mean(W_neto_horario) / 100 * 100  # %

print(f'Producción anual: {produccion_anual:.2f} GWh')
print(f'Factor de capacidad: {factor_capacidad:.2f}%')

# === 3.2.3) RELACIÓN ÁREA vs SM ===

SM_valores = np.linspace(1, 4, 20)  # SM entre 1 y 4 con 20 puntos
area_real = SM_valores * area       # Área real según SM (m²)

plt.figure(figsize=(8, 5))
plt.plot(SM_valores, area_real / 1e6, marker='o')
plt.xlabel('Múltiplo Solar (SM)')
plt.ylabel('Área real (millones de m²)')
plt.title('Área real del campo solar en función del Múltiplo Solar (SM)')
plt.grid(True)
plt.savefig('area_vs_SM.png')
plt.show()

# ======================================================================
# ========================== 3.3.1 TES =================================
# ======================================================================

print("\n=== 3.3.1 Inventario diario de energía almacenada TES ===")

# --- 1) Selección del día más soleado (máximo DNI acumulado)
dni_por_dia = []
for d in range(365):
    inicio = d * 24
    fin = inicio + 24
    dni_por_dia.append(np.sum(DNI_horario[inicio:fin]))

dia_soleado = np.argmax(dni_por_dia)
print(f"El día más soleado es el día: {dia_soleado+1}, DNI acumulado = {dni_por_dia[dia_soleado]:.2f} Wh/m²")

# --- 2) Extraer el Q_campo y W_neto de ese día (24 horas)
inicio = dia_soleado * 24
fin = inicio + 24

Qdia = Q_campo_horario[inicio:fin]        # MWt
Wdia = W_neto_horario[inicio:fin]         # MW eléctricos generados

# --- 3) Cálculo del balance térmico horario (solo excedentes cargan TES)
excedente_termico = np.maximum(Qdia - calor_PC, 0)   # MWt → carga TES
deficit_termico = np.maximum(calor_PC - Qdia, 0)     # MWt → descarga TES

# --- 4) Modelar inventario TES (variable capacidad = 8 y 18 horas)
def inventario_TES(cap_horas):
    cap_MWh = cap_horas * calor_PC   # Capacidad TES [MWh térmicos]
    inventario = [0]  # inicia descargado
    for h in range(24):
        E = inventario[-1] + excedente_termico[h] * 1  # carga por hora
        E -= deficit_termico[h] * 1                    # descarga por hora
        E = max(0, min(E, cap_MWh))                    # límites TES
        inventario.append(E)
    return np.array(inventario[1:]), cap_MWh

TES8, cap8 = inventario_TES(8)
TES18, cap18 = inventario_TES(18)

# --- 5) Gráfico comparativo para 8h y 18h
horas = np.arange(24)

# === Gráfico TES 8h ===
plt.figure(figsize=(10, 5))
plt.plot(horas, TES8, linewidth=2, color='steelblue')
plt.title(f'Inventario de energía TES (8h) - Día más soleado: {dia_soleado+1}')
plt.xlabel('Hora del día')
plt.ylabel('Energía almacenada (MWh térmicos)')
plt.grid(True)
plt.savefig('TES_8h.png')
plt.show()


# === Gráfico TES 18h ===
plt.figure(figsize=(10, 5))
plt.plot(horas, TES18, linewidth=2, color='darkorange')
plt.title(f'Inventario de energía TES (18h) - Día más soleado: {dia_soleado+1}')
plt.xlabel('Hora del día')
plt.ylabel('Energía almacenada (MWh térmicos)')
plt.grid(True)
plt.savefig('TES_18h.png')
plt.show()


print(f"Capacidad TES (8h): {cap8:.2f} MWh térmicos")
print(f"Capacidad TES (18h): {cap18:.2f} MWh térmicos")


# === 3.4.1 Despacho con TES (8 h y 18 h) ===

# ============================
# 3.4.1 Factor de capacidad vs SM (con TES 10 h)
# ============================

# Capacidad TES 10h
TES10_max = 10 * calor_PC  # MWh térmicos

# Valores de SM
SM_vals = np.linspace(1, 4, 6)
FC_vals = []  # Factor de capacidad de cada SM

def despacho_anual(Q_solar, TES_max):
    TES = 0.0
    W_net = []

    for Q in Q_solar:  # MW térmicos horario
        Q_req = calor_PC  # demanda para 100 MW netos

        if Q >= Q_req:  # Sobra energía → se carga TES
            excedente = Q - Q_req
            TES += excedente
            if TES > TES_max:
                TES = TES_max
            W_net.append(100)  # turbina a full
        else:
            deficit = Q_req - Q
            descarga = min(deficit, TES)
            TES -= descarga
            W = eficiencia_ciclo * (Q + descarga)
            W = min(W, 100)
            W_net.append(W)

    return np.array(W_net)

# Calcular FC para cada SM
for SM in SM_vals:
    Q_SM = Q_campo_horario * SM  # escala del campo solar
    W = despacho_anual(Q_SM, TES10_max)  # MW netos
    energia_anual_GWh = np.sum(W) / 1000  # GWh
    FC = np.mean(W) / 100 * 100  # %
    FC_vals.append(FC)

# ======== Gráfico 3.4.1 ========
plt.figure(figsize=(8,5))
plt.plot(SM_vals, FC_vals, marker='o', linestyle='-', color='purple')
plt.title('3.4.1 Factor de capacidad vs SM (TES 10 horas)')
plt.xlabel('Múltiplo Solar (SM)')
plt.ylabel('Factor de capacidad (%)')
plt.grid(True)
plt.savefig('3_4_1_FactorCapacidad_SM.png')
plt.show()

# Mostrar resultados en consola
#for sm, fc in zip(SM_vals, FC_vals):
   # print(f"SM={sm:.2f} → FC={fc:.2f}%")

# ===============================
# 3.4.2 Curvas de nivel FC vs SM y TES
# ===============================

# Valores de SM (más resolución que 3.4.1)
SM_vals = np.linspace(1, 4, 20)

# Valores de TES en horas
TES_horas = np.array([5, 10, 15, 20])

# Matriz de Factor de Capacidad FC [%]
FC_matrix = np.zeros((len(TES_horas), len(SM_vals)))

# Función anual de despacho (ya creada antes)
def despacho_anual(Q_solar, TES_max):
    TES = 0.0
    W_net = []

    for Q in Q_solar:
        Q_req = calor_PC

        if Q >= Q_req:
            excedente = Q - Q_req
            TES += excedente
            TES = min(TES, TES_max)
            W_net.append(100)
        else:
            deficit = Q_req - Q
            descarga = min(deficit, TES)
            TES -= descarga
            W = eficiencia_ciclo * (Q + descarga)
            W = min(W, 100)
            W_net.append(W)

    return np.array(W_net)

# Llenar matriz FC
for i, hrs in enumerate(TES_horas):
    TES_cap = hrs * calor_PC  # MWh térmicos

    for j, SM in enumerate(SM_vals):
        Q_SM = Q_campo_horario * SM
        W = despacho_anual(Q_SM, TES_cap)
        FC_matrix[i, j] = np.mean(W) / 100 * 100  # %

# ======== Gráfico de Contorno ========
plt.figure(figsize=(10,6))
X, Y = np.meshgrid(SM_vals, TES_horas)
cp = plt.contourf(X, Y, FC_matrix, levels=15, cmap='viridis')
plt.colorbar(cp, label='Factor de capacidad (%)')
plt.xlabel('Múltiplo Solar (SM)')
plt.ylabel('Almacenamiento térmico (horas)')
plt.title('3.4.2 Curvas de nivel FC (%) vs SM y tamaño TES')
plt.grid(False)
plt.savefig('3_4_2_FC_Contour.png')
plt.show()

# Mostrar matriz en consola (opcional)
#print("=== FC (%) para cada combinación SM - TES ===")
#for h, row in zip(TES_horas, FC_matrix):
#    print(f"{h}h TES →", ["{:.1f}".format(v) for v in row])

# ------------------------------
# Costos unitarios definidos
# ------------------------------
costo_campo_m2 = 170       # USD/m2 (valor típico LCOE CSP)
costo_TES_m3   = 700       # USD/m3 (tanque + aislamiento + sales)

# ------------------------------
# Parámetros físico–termodinámicos del TES
# ------------------------------
rho_avg = 0.5*( (2090 - 0.636*T_fria) + (2090 - 0.636*T_caliente) )        # kg/m3
cp_avg  = 0.5*( (1443 + 0.172*T_fria) + (1443 + 0.172*T_caliente) )       # J/kgK 
DeltaT_tes = T_caliente - T_fria                                           # K

# Conversión MWh → Joules
MWh_to_J = 3.6e9

# ------------------------------
# Rango de análisis
# ------------------------------
SM_vals = np.linspace(1,4,20)
TES_horas_vals = np.array([5,10,15,20])

# Matriz de resultados
cost_matrix = np.zeros((len(TES_horas_vals), len(SM_vals)))

# Área ya calculada previamente como: area = N * area_loop
# Para SM ≠ 1: área_real = SM * area
# (IMPORTANTE: "area" viene calculada en tu código original)

# ------------------------------
# Cálculo del costo del sistema
# ------------------------------
for i, hrs in enumerate(TES_horas_vals):

    # Energía almacenada (MWh térmicos)
    cap_MWh = hrs * calor_PC

    # Energía equivalente en J
    E_J = cap_MWh * MWh_to_J

    # Volumen TES requerido
    V_TES = E_J / (rho_avg * cp_avg * DeltaT_tes)

    # Costo TES
    costo_TES = V_TES * costo_TES_m3

    for j, SM in enumerate(SM_vals):

        # Área real del campo solar
        area_real = SM * area

        # Costo campo solar
        costo_campo = area_real * costo_campo_m2

        # Costo total
        cost_matrix[i,j] = costo_campo + costo_TES


# ------------------------------
# Gráfico tipo mapa de calor
# ------------------------------
plt.figure(figsize=(10,6))
X, Y = np.meshgrid(SM_vals, TES_horas_vals)

cp = plt.contourf(X, Y, cost_matrix/1e6, levels=20, cmap='viridis')
plt.colorbar(cp, label='Costo total (Millones USD)')

plt.xlabel('Múltiplo Solar (SM)')
plt.ylabel('Almacenamiento TES (horas)')
plt.title('3.4.3 Costo total del sistema (Campo solar + TES)')
plt.grid(False)
plt.savefig('3_4_3_Costo_Sistema.png', dpi=200)
plt.show()

