from tqdm import tqdm
import bitarray
from matplotlib import pyplot as plt
import numpy as np
from bitarray import bitarray
from random import randint
from numpy.linalg import svd
from scipy import signal
from statistics import mode
import seaborn as sns
import pandas as pd
import collections

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

a = ([1, -2, 2, -1.97, 1])
b = ([0, 0.0028, -0.002, -0.002, 0.0028])
r = 2
nlin = 65
ncol = 34
p_ini = 500
num_seeds=1000
u = prbs_11
y = signal.lfilter(b, a, u)

# ------------------------- Ruído -------------------------
SNR_dB = 20
# # Potência do sinal
Py = np.mean(y**2)
# # Potência do ruído
Pruido = Py / (10 ** (SNR_dB / 10))
# ruido = 0
dpadrao = np.sqrt(Pruido)


# ------------------------- Entrada -------------------------
resultados_criterio = []
r_otimos = []



for seed in range(1, num_seeds + 1):
    #Gerar Sinal e Saída
    prbs = prbs_sequence(10, seed)
    u = 2 * np.array(prbs.tolist(), dtype=float) - 1
    y = signal.lfilter(b, a, u)

    #Ruído
    ruido = np.random.normal(loc=0, scale=dpadrao, size=y.shape)
    y_ruido = y + ruido


    #Construir Matriz Ay e aplicar SVD
    Ay = np.zeros((nlin, ncol))
    for k in range(nlin):
        it = p_ini + k
        Ay[k, :] = -y_ruido[it - 1 : it - ncol - 1: -1]

    U, S, Vt = np.linalg.svd(Ay, full_matrices=False)
    Vr = Vt[:r, :].T  # Mantendo apenas as r colunas principais

    #Calcular a Variável "Criterio"
    R = Vr @ Vr.T
    coluna_1 = R[:, 0]
    criterio = 1 / np.abs(coluna_1)

    U, S, Vt = np.linalg.svd(Ay, full_matrices=False)

    p = len(S)
    erros = np.zeros(p)
    # Ignora os extremos
    erros[0] = np.inf
    erros[-1] = np.inf
    erros[1] = np.inf
    erros[-2] = np.inf

    # Testa todas as posições de corte 'q' possíveis (de 1 até 33)
    for q in range(1, p - 1):
        # Divide os valores em dois grupos no ponto 'q'
        grupo_sinal = S[:q]
        grupo_ruido = S[q:]

        # Calcula o quanto os valores variam dentro de cada grupo
        var_sinal = np.sum((grupo_sinal - np.mean(grupo_sinal)) ** 2)
        var_ruido = np.sum((grupo_ruido - np.mean(grupo_ruido)) ** 2)

        # O erro deste corte é a soma das variações
        erros[q] = var_sinal + var_ruido

    # O 'r' ideal é a posição que resultou no menor erro possível
    r_otimo = np.argmin(erros)
    r_otimos.append(r_otimo)


    #Armazenar os 34 valores para esta seed
    for i in range(ncol):
        resultados_criterio.append({
            'Atraso': i + 1 ,
            'Valor_Criterio': criterio[i],
            'Seed': seed
        })

#---------------------- BOX PLOTS ----------------------
np.set_printoptions(suppress=True, precision=6)

r = mode(r_otimos)

print(f"\n=> O ponto de corte ideal (fator r) calculado é: {r}")

frequencias = collections.Counter(r_otimos)
print(f"Frequência dos cortes encontrados nas {num_seeds} iterações: {frequencias}")


df_criterio = pd.DataFrame(resultados_criterio)

fig, ax = plt.subplots(figsize=(15, 6))
sns.boxplot(x='Atraso', y='Valor_Criterio', data=df_criterio)
ax.set_yscale('log')

ax.set_title("Distribution of the 'Criteria' Variable")
ax.set_ylabel("Criterion Value")
ax.set_xlabel("Lags")
plt.grid(True, which="both", ls="-", alpha=0.2)

# ---------------------- MÍNIMOS QUADRADOS ----------------------
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error, root_mean_squared_error
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.display_results import results
from sysidentpy.simulation import SimulateNARMAX

y_ruido = y + ruido
y_total = y_ruido.reshape(-1, 1)
u_total = u.reshape(-1, 1)
div = int(0.5*len(y_total))
y_train = y_total[div:]
u_train = u_total[div:]
y_vali = y_total[:div]
u_vali = u_total[:div]

ylags = [1, 16]
# ylags = [1, 13, 22, 26] #Sem Ruído
xlags = list(range(1, 35))
basis_function = Polynomial(degree=1)
estimator = LeastSquares(unbiased=True)
model = FROLS(
    order_selection=False,
    n_terms=37,
    ylag=ylags,
    xlag=xlags,
    basis_function=basis_function,
    model_type='NARMAX',
    estimator=estimator,
)

model.fit(X=u_train, y=y_train)

D= pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(D)
# print(f'\n Parâmetros do modelo: \n {model.theta}')

cte = ~np.all(model.final_model == 0, axis=1)
model.final_model = model.final_model[cte, :]
model.theta = model.theta[cte, :]
if model.err is not None and len(model.err) == len(cte):
    model.err = model.err[cte]
model.n_terms = model.final_model.shape[0]

full_model_code = model.final_model.copy()
reduced_model = full_model_code.copy()


# Reestima os parâmetros mantendo a estrutura fixa em reduced_model
simulator = SimulateNARMAX(
    estimator=model.estimator,
    elag=model.elag,
    estimate_parameter=True,
    model_type=model.model_type,
    basis_function=model.basis_function,
)

_ = simulator.simulate(
    X_train=u_train,
    y_train=y_train,
    X_test=u_train,
    y_test=y_train,
    model_code=reduced_model,
)

# Resultado final (modelo + parâmetros, após estimativa)
F = pd.DataFrame(
    results(
        simulator.final_model,
        simulator.theta,
        simulator.err,
        simulator.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print("Modelo sem a constante:\n",F)
# print(f'\n Parâmetros do modelo: \n {simulator.theta}')
# print(np.array(simulator.final_model))

yhat = model.predict(X=u_vali, y=y_vali)

rrse = root_relative_squared_error(y_vali, yhat)
print(f"\nErro RRSE: {rrse}")
rmse = root_mean_squared_error(y_vali, yhat)
print(f"RMSE: {rmse}")


# ---------------------- RAÍZES ----------------------
raizes_eq16 = np.roots(a)
magnitudes_eq16 = np.abs(raizes_eq16)

print("\n--- Pólos da Planta Original (Equação 16) ---")
for i, (raiz, mag) in enumerate(zip(raizes_eq16, magnitudes_eq16)):
    print(f"Pólo {i+1:02d}: {raiz:+.4f} | Magnitude: {mag:.4f}")

# ---------------------- PLOTS ----------------------

plt.figure(figsize=(14, 5))
plt.plot(y_vali[:1000], label='Real data', color='black')
plt.plot(yhat[:1000], label='SysIdentPy Model', color='red', linestyle='--')
plt.title("Validation of the estimated model")
plt.legend()
plt.grid(True)
plt.show()


# print("\n--- Valores Singulares (S) Saída ---")
# print(S[:8])


amp_sinal = np.max(y)
amp_ruido = np.max(ruido)

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