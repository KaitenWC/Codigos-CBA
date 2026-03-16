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


Tf = 3.1e-3
Ts = Tf/4

def prbs_sequence(prbs_bits:int, rng_seed:int) -> bitarray:
    """Gera uma sequência de int do tipo PRBS

    Args:
        prbs_bits (int): Quantidade de bits do gerador PRBS
        rng_seed (int): Valor inicial do gerador PRBS

    Returns:
        bitarray: Sinal PRBS
    """
    prbs_types = {
        3: {'bit_1':2 , 'bit_2':1 }, #size = 7
        4: {'bit_1':3 , 'bit_2':2 }, #size = 15
        5: {'bit_1':4 , 'bit_2':2 }, #size = 31
        6: {'bit_1':5 , 'bit_2':4 }, #size = 63
        7: {'bit_1':6 , 'bit_2':5 }, #size = 127
        9: {'bit_1':8 , 'bit_2':4 }, #size = 511
       10: {'bit_1':9 , 'bit_2':6 }, #size = 1_023
       11: {'bit_1':10, 'bit_2':8 }, #size = 2_047
       15: {'bit_1':14, 'bit_2':13}, #size = 32_767
       17: {'bit_1':16, 'bit_2':13}, #size = 131_071
       18: {'bit_1':17, 'bit_2':10}, #size = 262_143
       20: {'bit_1':19, 'bit_2':16}, #size = 1_048_575
       21: {'bit_1':20, 'bit_2':18}, #size = 2_097_151
       22: {'bit_1':21, 'bit_2':20}, #size = 4_194_303
       23: {'bit_1':22, 'bit_2':17}, #size = 8_388_607
    #  25: {'bit_1':24, 'bit_2':21}, #size = 33_554_431
    #  28: {'bit_1':27, 'bit_2':24}, #size = 268_435_455
    #  29: {'bit_1':28, 'bit_2':26}, #size = 536_870_911
    #  31: {'bit_1':30, 'bit_2':27}, #size = 2_147_483_647
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
    for _ in tqdm(range(size-1), desc=f'Gerando sinal (PRBS{prbs_bits:d})'):
        new_bit = ~((new_value>>bit_1) ^ (new_value>>bit_2)) & 0x1
        new_value = ((new_value<<1) + new_bit) & size
        #Fechou um período ou atingiu estado proibido: retorna o resultado
        if (new_value == start_value):
            return bit_sequence
        bit_sequence.append(bool(new_bit))
    return bit_sequence


prbs = prbs_sequence(10,100)
prbs_01 = np.array(prbs.tolist(), dtype=float)
prbs_11 = 2 * prbs_01 - 1

a = ([1,-2,2,-1.97,1])
b = ([0, 0.0028,-0.002,-0.002,0.0028])
tempo = np.arange(len(prbs_11)) * Ts
nlin = 65
ncol = 34
p_ini = 100
Ay = np.zeros((nlin,ncol))
u = prbs_11
y = signal.lfilter(b, a, u)
Au = np.zeros((nlin,ncol))

# ------------------------- Ruído -------------------------
SNR_dB = 20
#
# # Potência do sinal
Py = np.var(y[34:100])
#
# # Variância do ruído
ruido_var = Py / (10 ** (SNR_dB / 10))
# # np.random.seed(10) #Definindo um ruído fixo para todas as simulações
# # Ruído branco Gaussiano
# # ruido = np.sqrt(ruido_var) * np.random.randn(len(y))
# ruido = 0

# ------------------------- Entrada -------------------------
num_seeds = 100
nlin = 65
ncol = 34
p_ini = 100
r = 4
a = [1, -2, 2, -1.97, 1]
b = [0, 0.0028, -0.002, -0.002, 0.0028]

resultados_criterio = []
S_acumulado = []



for seed in range(1, num_seeds + 1):
    #Gerar Sinal e Saída
    prbs = prbs_sequence(10, seed)  # Sua função de PRBS
    u = 2 * np.array(prbs.tolist(), dtype=float) - 1
    y = signal.lfilter(b, a, u)

    #Ruído
    np.random.seed(seed)
    ruido = np.sqrt(ruido_var) * np.random.randn(len(y))
    y_ruido = y + ruido


    #Construir Matriz Ay e aplicar SVD
    Ay = np.zeros((nlin, ncol))
    for k in range(nlin):
        it = p_ini + k
        Ay[k, :] = -y_ruido[it - 1: it - 1 - ncol: -1]

    U, S, Vt = np.linalg.svd(Ay, full_matrices=False)
    Vr = Vt[:r, :].T  # Mantendo apenas as r colunas principais

    #Calcular a Variável "Criterio"
    R = Vr @ Vr.T
    coluna_1 = R[:, 0]
    criterio = 1 / np.abs(coluna_1)

    U, S, Vt = np.linalg.svd(Ay, full_matrices=False)
    S_acumulado.append(S)

    #Armazenar os 34 valores para esta seed
    for i in range(ncol):
        resultados_criterio.append({
            'Atraso': i + 1,
            'Valor_Criterio': criterio[i],
            'Seed': seed
        })

#---------------------- VIOLIN PLOTS ----------------------
S_medio = np.mean(S_acumulado, axis=0)

print("Valores singulares médios:", S_medio)



df_criterio = pd.DataFrame(resultados_criterio)

fig, ax = plt.subplots(figsize=(15, 6))
# Usamos escala logarítmica no Y pois os picos de critério podem ser muito altos
sns.violinplot(x='Atraso', y='Valor_Criterio', data=df_criterio, palette='magma')
ax.set_yscale('log')

ax.set_title("Distribution of the 'Criteria' Variable - 100 Seeds")
ax.set_ylabel("Criterion Value")
ax.set_xlabel("Lags")
plt.grid(True, which="both", ls="-", alpha=0.2)

# ---------------------- MÍNIMOS QUADRADOS ----------------------
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error, root_mean_squared_error

y_ruido = y + ruido
y_total = y_ruido.reshape(-1, 1)
u_total = u.reshape(-1, 1)
div = int(0.5*len(y_total))
y_train = y_total[div:]
u_train = u_total[div:]
y_vali = y_total[:div]
u_vali = u_total[:div]

# ylags = [1, 13, 22, 26]
ylags = [1, 4, 13, 22]
xlags = list(range(1, 35))
basis_function = Polynomial(degree=1)
model = FROLS(
    order_selection=False,
    n_terms=35,
    ylag=ylags,
    xlag=xlags,
    basis_function=basis_function,
    model_type='NARMAX',
    info_criteria='bic',
)

model.fit(X=u_train, y=y_train)
print("\n--- Modelo Identificado ---")
# Mostra os atrasos escolhidos e seus coeficientes
Cte= ~np.all(model.final_model == 0, axis=1)


model.final_model = model.final_model[Cte]
model.theta = model.theta[Cte]


yhat = model.predict(X=u_vali, y=y_vali)


for i, theta in enumerate(model.theta):
    print(f"Termo {i+1}: {model.final_model[i]}  |  Coeficiente: {theta[0]:.4f}")


rrse = root_relative_squared_error(y_vali, yhat)
print(f"\nErro RRSE: {rrse:.5f}")
rmse = root_mean_squared_error(y_vali, yhat)

print(f"RMSE: {rmse}")




# ---------------------- RAÍZES ----------------------
A_z = np.zeros(max(ylags) + 1)
A_z[0] = 1.0

# 2. Preenche os coeficientes buscando no modelo
for i, regressor in enumerate(model.final_model):
    # Filtra: tem que ser linear (tamanho 1) e ser da saída 'y' (código entre 1000 e 1999)
    if regressor[0] < 2000:
        lag = regressor[0] % 1000  # Descobre qual é o atraso (ex: 1002 vira 2)
        A_z[lag] = -model.theta[i][0]  # Inverte o sinal e guarda no polinômio
# 2. Calcular as raízes (pólos) e suas magnitudes
raizes = np.roots(A_z)
magnitudes = np.abs(raizes)

print("\n--- Pólos Estimados (Raízes do Denominador) ---")
for i, (raiz, mag) in enumerate(zip(raizes, magnitudes)):
    print(f"Pólo {i+1:02d}: {raiz:+.4f} | Magnitude: {mag:.4f}")
# ---------------------- PLOTS ----------------------

plt.figure(figsize=(14, 5))
plt.plot(y_vali[:200], label='Real data', color='black')
plt.plot(yhat[:200], label='SysIdentPy Model', color='red', linestyle='--')
plt.title("Validation of the estimated model")
plt.legend()
plt.grid(True)
plt.show()


print("\n--- Valores Singulares (S) Saída ---")
print(S[:8])


amp_sinal = max(y)
amp_ruido = max(ruido)

print("Sinal", amp_sinal)
print("Ruído", amp_ruido)



# plt.figure(figsize=(12, 5))
# plt.step(tempo[:100], prbs_11[:100], where='post')
# plt.title(f"Sinal PRBS (Primeiras 100 amostras) - Ts = {Ts:.2e}s")
# plt.xlabel("Tempo (s)")
# plt.ylabel("Amplitude")
# plt.grid(True, which='both', linestyle='--')
# plt.yticks([-1, 0, 1]) # Força mostrar apenas esses valores no eixo Y
# plt.show()