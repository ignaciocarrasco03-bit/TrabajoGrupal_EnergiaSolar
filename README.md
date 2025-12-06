# TrabajoGrupal_EnergiaSolar
Código cálculos lenguaje python

import math

eta_cycle = 0.4693
W_net = 100  # MW
Q_PC = W_net / eta_cycle  # MW
print(f'Q_PC: {Q_PC:.3f} MW')

a = 1443
b = 0.172
Tc = 239
Th = 600
DeltaT = Th - Tc
diff_sq = Th**2 - Tc**2
Delta_H = a * DeltaT + (b / 2) * diff_sq  # J/kg
print(f'Delta_H: {Delta_H:.2f} J/kg')

m_total = (Q_PC * 1e6) / Delta_H  # kg/s
print(f'm_total: {m_total:.2f} kg/s')

# Campo solar
A_loop = 574.77  # m2
Am = 47.89  # m2 per module
n_modules = 12
IAM = 0.9909
eta0 = 0.84
eta_mi = 0.97
opt = IAM * eta0 * eta_mi
print(f'opt: {opt:.4f}')

DNI = 1040  # W/m2
T_amb = 19.5

def simulate_T_end(m_loop):
    T = Tc
    for _ in range(n_modules):
        dT = T - T_amb
        loss1 = 0.15 * dT / DNI  # corregido
        loss2 = 0.0016 * dT**2 / DNI
        eta = opt - loss1 - loss2
        if eta < 0:
            eta = 0
        dQ = eta * DNI * Am  # W
        Cp = 1443 + 0.172 * T  # J/kg°C
        dTemp = dQ / (m_loop * Cp) if m_loop > 0 else 0
        T += dTemp
    return T

# Búsqueda binaria para N mínimo
low = 1
high = 10000
while low < high:
    mid = (low + high) // 2
    m_l = m_total / mid
    T_end = simulate_T_end(m_l)
    if T_end >= Th:
        high = mid
    else:
        low = mid + 1
N = low
print(f'Number of loops: {N}')
area = N * A_loop
print(f'Associated area: {area:.2f} m2')
T_final = simulate_T_end(m_total / N)
print(f'T_end for N: {T_final:.2f} °C')
