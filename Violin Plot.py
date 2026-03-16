from tqdm import tqdm
import bitarray
from matplotlib import pyplot as plt
import numpy as np
from bitarray import bitarray
from random import randint
from numpy.linalg import svd
from scipy import signal
import seaborn as sns
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error, root_mean_squared_error

Tf = 3.1e-3
Ts = Tf/4

def prbs_sequence(prbs_bits:int, rng_seed:int) -> bitarray:
    prbs_types = {
        3: {'bit_1':2 , 'bit_2':1 },
        4: {'bit_1':3 , 'bit_2':2 },
        5: {'bit_1':4 , 'bit_2':2 },
        6: {'bit_1':5 , 'bit_2':4 },
        7: {'bit_1':6 , 'bit_2':5 },
        9: {'bit_1':8 , 'bit_2':4 },
       10: {'bit_1':9 , 'bit_2':6 },
       11: {'bit_1':10, 'bit_2':8 },
       15: {'bit_1':14, 'bit_2':13},
       17: {'bit_1':16, 'bit_2':13},
       18: {'bit_1':17, 'bit_2':10},
       20: {'bit_1':19, 'bit_2':16},
       21: {'bit_1':20, 'bit_2':18},
       22: {'bit_1':21, 'bit_2':20},
       23: {'bit_1':22, 'bit_2':17},
    }
    if prbs_bits >= max(prbs_types.keys()):
        prbs_bits = max(prbs_types.keys())
    else:
        prbs_bits = min(b for b in prbs_types.keys() if b >= prbs_bits)
    size = (2**prbs_bits) - 1
    bit_1 = prbs_types[prbs_bits]['bit_1']
    bit_2 = prbs_types[prbs_bits]['bit_2']
    start_value = randint(0,size-1) if rng_seed is None else rng_seed
    start_value = int(min(max(start_value, 0), size-1))

    bit_sequence = bitarray([start_value & 0x1])
    new_value = start_value
    for _ in range(size-1):
        new_bit = ~((new_value>>bit_1) ^ (new_value>>bit_2)) & 0x1
        new_value = ((new_value<<1) + new_bit) & size
        if (new_value == start_value):
            return bit_sequence
        bit_sequence.append(bool(new_bit))
    return bit_sequence

# Inicialização e cálculo da variância do ruído (mantendo lógica original)
prbs = prbs_sequence(10,100)
prbs_01 = np.array(prbs.tolist(), dtype=float)
prbs_11 = 2 * prbs_01 - 1
a = ([1,-2,2,-1.97,1])
b = ([0, 0.0028,-0.002,-0.002,0.0028])
tempo = np.arange(len(prbs_11)) * Ts
u_init = prbs_11
y_init = signal.lfilter(b, a, u_init)
SNR_dB = 20
Py = np.var(y_init[34:100])
ruido_var = Py / (10 ** (SNR_dB / 10))

# Configurações da simulação
num_seeds = 100
nlin = 65
ncol = 34
p_ini = 100
r = 4
a = [1, -2, 2, -1.97, 1]
b = [0, 0.0028, -0.002, -0.002, 0.0028]

resultados_criterio_limpo = []
resultados_criterio_ruido = []
S_acumulado_limpo = []
S_acumulado_ruido = []

# Variáveis para armazenar o último sinal gerado para uso no SysIdentPy
last_u = None
last_y_limpo = None
last_y_ruido = None

for seed in tqdm(range(1, num_seeds + 1), desc="Simulando Seeds"):
    # Gerar Sinal e Saída
    prbs = prbs_sequence(10, seed)
    u = 2 * np.array(prbs.tolist(), dtype=float) - 1
    y_limpo = signal.lfilter(b, a, u)

    # Gerar Ruído
    np.random.seed(seed)
    ruido = np.sqrt(ruido_var) * np.random.randn(len(y_limpo))
    y_ruido = y_limpo + ruido

    last_u = u
    last_y_limpo = y_limpo
    last_y_ruido = y_ruido

    # --- Processamento SEM Ruído ---
    Ay_limpo = np.zeros((nlin, ncol))
    for k in range(nlin):
        it = p_ini + k
        Ay_limpo[k, :] = -y_limpo[it - 1: it - 1 - ncol: -1]

    U, S, Vt = np.linalg.svd(Ay_limpo, full_matrices=False)
    Vr = Vt[:r, :].T
    R = Vr @ Vr.T
    coluna_1 = R[:, 0]
    criterio_limpo = 1 / np.abs(coluna_1)
    S_acumulado_limpo.append(S)

    for i in range(ncol):
        resultados_criterio_limpo.append({
            'Atraso': i + 1,
            'Valor_Criterio': criterio_limpo[i],
            'Seed': seed
        })

    # --- Processamento COM Ruído ---
    Ay_ruido = np.zeros((nlin, ncol))
    for k in range(nlin):
        it = p_ini + k
        Ay_ruido[k, :] = -y_ruido[it - 1: it - 1 - ncol: -1]

    U, S, Vt = np.linalg.svd(Ay_ruido, full_matrices=False)
    Vr = Vt[:r, :].T
    R = Vr @ Vr.T
    coluna_1 = R[:, 0]
    criterio_ruido = 1 / np.abs(coluna_1)
    S_acumulado_ruido.append(S)

    for i in range(ncol):
        resultados_criterio_ruido.append({
            'Atraso': i + 1,
            'Valor_Criterio': criterio_ruido[i],
            'Seed': seed
        })

# ---------------------- VIOLIN PLOTS ----------------------
S_medio_limpo = np.mean(S_acumulado_limpo, axis=0)
S_medio_ruido = np.mean(S_acumulado_ruido, axis=0)

print("Valores singulares médios (Sem Ruído):", S_medio_limpo)
print("Valores singulares médios (Com Ruído):", S_medio_ruido)

df_criterio_limpo = pd.DataFrame(resultados_criterio_limpo)
df_criterio_ruido = pd.DataFrame(resultados_criterio_ruido)

fig, ax = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# Plot Sem Ruído
sns.violinplot(x='Atraso', y='Valor_Criterio', data=df_criterio_limpo, palette='magma', ax=ax[0])
ax[0].set_yscale('log')
ax[0].set_title("Distribuição do Critério (Sem Ruído) - 100 Seeds")
ax[0].set_ylabel("Valor do Critério (Escala Log)")
ax[0].grid(True, which="both", ls="-", alpha=0.2)

# Plot Com Ruído
P_Sinal_Calc = np.var(last_y_limpo) # Usando o ultimo sinal como referencia de potência
P_Ruido_Calc = ruido_var
sns.violinplot(x='Atraso', y='Valor_Criterio', data=df_criterio_ruido, palette='magma', ax=ax[1])
ax[1].set_yscale('log')
ax[1].set_title(f"Distribuição do Critério (Com Ruído) - 100 Seeds | P_Sinal: {P_Sinal_Calc:.2e} | P_Ruído: {P_Ruido_Calc:.2e}")
ax[1].set_ylabel("Valor do Critério (Escala Log)")
ax[1].set_xlabel("Índice da Amostra (Atraso)")
ax[1].grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()

# ---------------------- MÍNIMOS QUADRADOS ----------------------
ylags = [1, 4, 13, 26]
# ylags = [1, 4, 22, 26]
xlags = list(range(1, 34))
basis_function = Polynomial(degree=1)

# ==============================================================================
# CENÁRIO 1: SEM RUÍDO
# ==============================================================================
print("\n" + "="*40)
print("CENÁRIO: SEM RUÍDO")
print("="*40)

y_total = last_y_limpo.reshape(-1, 1)
u_total = last_u.reshape(-1, 1)
div = int(0.5*len(y_total))
y_train = y_total[div:]
u_train = u_total[div:]
y_vali = y_total[:div]
u_vali = u_total[:div]

model_clean = FROLS(
    order_selection=False,
    n_terms=34,
    ylag=ylags,
    xlag=xlags,
    basis_function=basis_function,
    model_type='NARMAX',
    info_criteria='bic',
)

model_clean.fit(X=u_train, y=y_train)
print("\n--- Modelo Identificado (Sem Ruído) ---")
Cte = ~np.all(model_clean.final_model == 0, axis=1)
model_clean.final_model = model_clean.final_model[Cte]
model_clean.theta = model_clean.theta[Cte]

poly_str = "y(k) = "
for i, theta in enumerate(model_clean.theta):
    print(f"Termo {i+1}: {model_clean.final_model[i]}  |  Coeficiente: {theta[0]:.4f}")
    term_str = f"({theta[0]:.4f}) * Termo_{i+1}"
    if i > 0:
        poly_str += " + "
    poly_str += term_str

print(f"\nFormato Polinomial: {poly_str}")

yhat_clean = model_clean.predict(X=u_vali, y=y_vali)
rrse_clean = root_relative_squared_error(y_vali, yhat_clean)
print(f"\nErro RRSE: {rrse_clean:.5f}")
rmse_clean = root_mean_squared_error(y_vali, yhat_clean)
print(f"RMSE: {rmse_clean}")

plt.figure(figsize=(14, 5))
plt.plot(y_vali[:200], label='Dados Reais (Simulado - Sem Ruído)', color='black')
plt.plot(yhat_clean[:200], label='Modelo SysIdentPy', color='blue', linestyle='--')
plt.title("Validação do Modelo Estimado (Sem Ruído)")
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Valores Singulares (S) Saída (Sem Ruído) ---")
print(S_medio_limpo[:8])
print("Sinal (Max):", np.max(last_y_limpo))
print("Ruído (Max):", 0)


# ==============================================================================
# CENÁRIO 2: COM RUÍDO
# ==============================================================================
print("\n" + "="*40)
print("CENÁRIO: COM RUÍDO")
print("="*40)

y_total = last_y_ruido.reshape(-1, 1)
u_total = last_u.reshape(-1, 1)
div = int(0.5*len(y_total))
y_train = y_total[div:]
u_train = u_total[div:]
y_vali = y_total[:div]
u_vali = u_total[:div]

model_noisy = FROLS(
    order_selection=False,
    n_terms=34,
    ylag=ylags,
    xlag=xlags,
    basis_function=basis_function,
    model_type='NARMAX',
    info_criteria='bic',
)

model_noisy.fit(X=u_train, y=y_train)
print("\n--- Modelo Identificado (Com Ruído) ---")
Cte = ~np.all(model_noisy.final_model == 0, axis=1)
model_noisy.final_model = model_noisy.final_model[Cte]
model_noisy.theta = model_noisy.theta[Cte]

poly_str = "y(k) = "
for i, theta in enumerate(model_noisy.theta):
    print(f"Termo {i+1}: {model_noisy.final_model[i]}  |  Coeficiente: {theta[0]:.4f}")
    term_str = f"({theta[0]:.4f}) * Termo_{i+1}"
    if i > 0:
        poly_str += " + "
    poly_str += term_str

print(f"\nFormato Polinomial: {poly_str}")

yhat_noisy = model_noisy.predict(X=u_vali, y=y_vali)
rrse_noisy = root_relative_squared_error(y_vali, yhat_noisy)
print(f"\nErro RRSE: {rrse_noisy:.5f}")
rmse_noisy = root_mean_squared_error(y_vali, yhat_noisy)
print(f"RMSE: {rmse_noisy}")

plt.figure(figsize=(14, 5))
plt.plot(y_vali[:200], label='Dados Reais (Simulado - Com Ruído)', color='black')
plt.plot(yhat_noisy[:200], label='Modelo SysIdentPy', color='red', linestyle='--')
plt.title("Validação do Modelo Estimado (Com Ruído)")
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Valores Singulares (S) Saída (Com Ruído) ---")
print(S_medio_ruido[:8])
print("Sinal (Max):", np.max(last_y_ruido))
print("Ruído (Max):", np.max(last_y_ruido - last_y_limpo))