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

prbs = prbs_sequence(10,20)
prbs_01 = np.array(prbs.tolist(), dtype=float)
prbs_11 = 2 * prbs_01 - 1
u = prbs_01


# ------------------------- Entrada -------------------------
num_seeds = 100
nlin = 65
ncol = 34
p_ini = 100
r = 4
a = [1.0, -1.7649, 0.8027]
b = [0.0, -0.73578, 0.075129, 0.8661]
y = signal.lfilter(b, a, u)



resultados_criterio = []
S_acumulado = []

# ------------------------- Ruído -------------------------
SNR_dB = 20
# # Potência do sinal
Py = np.mean(y**2)
# # Potência do ruído
Pruido = Py / (10 ** (SNR_dB / 10))
# ruido = 0
dpadrao = np.sqrt(Pruido)
ruido = np.random.normal(loc=0, scale=dpadrao, size=len(y))

# ------------------------- Resposta -------------------------
for seed in range(1, num_seeds + 1):
    # Gerar Sinal e Saída
    prbs = prbs_sequence(10, seed)  #PRBS
    u = 2 * np.array(prbs.tolist(), dtype=float) - 1
    y = signal.lfilter(b, a, u)


    #Ruído
    y_ruido = y + ruido


    #Construir Matriz Ay e aplicar SVD
    Ay = np.zeros((nlin, ncol))
    for k in range(nlin):
        it = p_ini + k
        Ay[k, :] = -y_ruido[it - 1: it - ncol - 1: -1]

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


S_medio = np.mean(S_acumulado, axis=0)

print("Valores singulares médios:", S_medio)
df_criterio = pd.DataFrame(resultados_criterio)


p = len(S_medio)
erros = np.zeros(p)
# Ignora os extremos
erros[0] = np.inf
erros[-1] = np.inf

# Testa todas as posições de corte 'q' possíveis
for q in range(1, p - 1):
    # Divide os valores em dois grupos no ponto 'q'
    grupo_sinal = S_medio[:q]
    grupo_ruido = S_medio[q:]

    # Calcula o quanto os valores variam dentro de cada grupo
    var_sinal = np.sum((grupo_sinal - np.mean(grupo_sinal)) ** 2)
    var_ruido = np.sum((grupo_ruido - np.mean(grupo_ruido)) ** 2)

    # O erro deste corte é a soma das variações
    erros[q] = var_sinal + var_ruido


r_otimo = np.argmin(erros)

print(f"\n=> O ponto de corte ideal (fator r) calculado é: {r_otimo}")

# ------------------------- Sysidentpy -------------------------
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


ylags = [1, 2, 7, 10]
# ylags = [1, 7, 32, 33]
xlags = list(range(1, 35))
basis_function = Polynomial(degree=1)
estimator = LeastSquares(unbiased=True)
model = FROLS(
    order_selection=False,
    n_terms=39,
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
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(D)
print(f'\n Parâmetros do modelo: \n {model.theta}')

cte = ~np.all(model.final_model == 0, axis=1)
model.final_model = model.final_model[cte, :]
model.theta = model.theta[cte, :]
if model.err is not None and len(model.err) == len(cte):
    model.err = model.err[cte]
model.n_terms = model.final_model.shape[0]

full_model_code = model.final_model.copy()
reduced_model = full_model_code.copy()

# print(model.final)

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
r = pd.DataFrame(
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
print(r)
print(f'\n Parâmetros do modelo: \n {simulator.theta}')
print(np.array(simulator.final_model))


yhat = model.predict(X=u_vali, y=y_vali)

rrse = root_relative_squared_error(y_vali, yhat)
print(f"\nErro RRSE: {rrse}")
rmse = root_mean_squared_error(y_vali, yhat)
print(f"RMSE: {rmse}")

# ------------------------- Plots -------------------------


for i, theta in enumerate(model.theta):
    print(f"Termo {i+1}: {model.final_model[i]}  |  Coeficiente: {theta[0]:.4f}")

yhat = model.predict(X=u_vali, y=y_vali)
rrse = root_relative_squared_error(y_vali, yhat)
print(f"\nErro RRSE: {rrse:.5f}")
rmse = root_mean_squared_error(y_vali, yhat)

print(f"RMSE: {rmse}")

plt.figure(figsize=(14, 5))
plt.plot(y_vali[:200], label='Dados Reais (Simulado)', color='black')
plt.plot(yhat[:200], label='Modelo SysIdentPy (SVD Lags)', color='red', linestyle='--')
plt.title("Validação do Modelo Estimado")
plt.legend()
plt.grid(True)


fig, ax = plt.subplots(figsize=(15, 6))
sns.boxplot(x='Atraso', y='Valor_Criterio', data=df_criterio, palette='magma')
ax.set_yscale('log')

ax.set_title("Distribuição da Variável 'Criterio' (SVD Pre-Identificação)")
ax.set_ylabel("Valor do Critério (Escala Log)")
ax.set_xlabel("Índice da Amostra (Atraso)")
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()
