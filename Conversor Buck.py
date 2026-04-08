from matplotlib import pyplot as plt
import numpy as np
from random import randint
from numpy.linalg import svd
import seaborn as sns
import pandas as pd

df_id = pd.read_csv(r'C:\Users\wende\OneDrive\Documentos\IC\Códigos\Fator de decimação\Dados Buck\buck_id.csv')
u_train = df_id['input'].values  # Coluna de entrada
y_train = df_id['y'].values      # Coluna de saída (y)

# Carregar os dados de validação
df_valid = pd.read_csv(r'C:\Users\wende\OneDrive\Documentos\IC\Códigos\Fator de decimação\Dados Buck\buck_valid.csv')
u_vali = df_valid['input'].values
y_vali = df_valid['y'].values

print(f"Dados de treino carregados: {len(y_train)} amostras")
print(f"Dados de validação carregados: {len(y_vali)} amostras")



nlin = 50
ncol = 34
p_ini = 50
r = 3
Ay = np.zeros((nlin,ncol))
u = u_train
y = y_train

# for k in range(nlin):
#     it = p_ini + k
#     # Ay[k, :] = -np.flip(y[it - ncol: it])
#     Ay[k, :] = -np.flip(y[it - ncol: it, 0])

for k in range(nlin):
    it = p_ini + k
    Ay[k, :] = -y[it - 1: it - 1 - ncol: -1]

U, S, Vt = np.linalg.svd(Ay, full_matrices=False)
Vr = Vt[:r, :].T  # Mantendo apenas as r colunas principais

# Calcular a Variável "Criterio"
R = Vr @ Vr.T
coluna_1 = R[:, 0]
criterio = 1 / np.abs(coluna_1)


print("Valores singulares:", S)



from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error, root_mean_squared_error


# y_ruido = y + ruido
u_treino = u_train.reshape(-1,1)
y_treino = y_train.reshape(-1,1)
u_valido = u_vali.reshape(-1,1)
y_valido = y_vali.reshape(-1,1)

ylags = [1, 15, 30]
xlags = list(range(1, 35))
# xlags = [1, 2]
basis_function = Polynomial(degree=1)
model = FROLS(
    order_selection=False,
    n_terms=37,
    ylag=ylags,
    xlag=xlags,
    basis_function=basis_function,
    model_type='NARMAX',
    info_criteria='bic',
)

model.fit(X=u_treino, y=y_treino)
print("\n--- Modelo Identificado ---")
# Mostra os atrasos escolhidos e seus coeficientes
yhat = model.predict(X=u_valido, y=y_valido)
Cte= ~np.all(model.final_model == 0, axis=1)
model.final_model = model.final_model[Cte]
model.theta = model.theta[Cte]

for i, theta in enumerate(model.theta):
    print(f"Termo {i+1}: {model.final_model[i]}  |  Coeficiente: {theta[0]:.4f}")


rrse = root_relative_squared_error(y_valido, yhat)
print(f"\nErro RRSE: {rrse:.5f}")
rmse = root_mean_squared_error(y_valido, yhat)

print(f"RMSE: {rmse}")

plt.figure(figsize=(14, 5))
plt.plot(y_vali[:1000], label='Dados Reais (Simulado)', color='black')
plt.plot(yhat[:1000], label='Modelo SysIdentPy (SVD Lags)', color='red', linestyle='--')
plt.title("Validação do Modelo Estimado")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(14,6))
plt.stem(criterio)
plt.xlabel("Índice da amostra")
plt.ylabel(r"$1/|R(i,1)|$")
plt.title("Critério para escolha do espaçamento ótimo")
plt.xticks(np.arange(1, ncol + 1, step=1))
plt.grid(True)
plt.show()

