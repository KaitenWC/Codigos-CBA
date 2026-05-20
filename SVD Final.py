from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from scipy import signal
import pandas as pd
import collections
from prbs_sequence import prbs_sequence
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error, root_mean_squared_error
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.display_results import results
from sysidentpy.simulation import SimulateNARMAX

# ------------------------- Seleção do Modelo -------------------------


# Basta comentar uma das 2 próximas linhas e selecionar qual o modelo
escolha = 'Buck'
# escolha = 'Artigo'


# Para plot do box lá embaixo
if escolha == 'Buck':
    nome_sistema = 'Conversor Buck'
else:
    nome_sistema = 'Sistema Linear (Benchmark)'


# Configurações baseadas na escolha
configuracoes_modelo = {
    'Buck': {
        'a': [1.0, -1.7649, 0.8027],
        'b': [0.0, -0.73578, 0.075129, 0.8661],
        'r': 10, # r possíveis após 100_000 simulações: r = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        # r2 =
        'ylags': [1, 8, 9, 11, 14, 15, 17, 18, 21, 29] # ylags possíveis (em ordem) a depender do r escolhido (100_000 simulacoes): r_6 = [1, 19, 7, 14, 24, 30]
    },
    'Artigo': {
        'a': [1.0, -1.954298, 1.923316, -1.939057, 0.989208],
        'b': [0.0, 0.020338, -0.029465, 0.029332, -0.020155],
        'r':7, # r possíveis após 100_000 simulações: r = [2, 3, 4, 5, 6, 7]
        'ylags': [1, 8, 13, 18, 24, 28, 32] # ylags possíveis (em ordem) a depender do r escolhido (100_000 simulacoes): r_2 = [1, 18, 17, 16, 19, 15, 20]
    }, # r_3 = [1, 20, 29], r_4 = [1, 14, 20, 29], r_5 = [1, 7, 16, 24, 28], r_6 = [1, 7, 13, 22, 26, 33], r_7 = [1, 8, 13, 18, 24, 28, 32]
}
for k in configuracoes_modelo:
    configuracoes_modelo[k]['n_terms'] = 34 + configuracoes_modelo[k]['r'] + 1


configuracao = configuracoes_modelo[escolha]
a = configuracao['a']
b = configuracao['b']
r = configuracao['r']
ylags = configuracao['ylags']
n_terms = configuracao['n_terms']

# Variáveis globais compartilhadas
nlin = 65
ncol = 34
p_ini = 100
SNR_dB = 20
# Semente para gerar a potência do sinal
seed = 0
num_seeds = 100_000

# ------------------------- Inicialização -------------------------
prbs = prbs_sequence(10, seed)  # Valores sobre 1 única seed para gerar a potência do ruído
prbs_01 = np.array(prbs.tolist(), dtype=float)
prbs_11 = 2 * prbs_01 - 1
u = prbs_11
y = signal.lfilter(b, a, u)


# ------------------------- Simulação de Múltiplas Sementes -------------------------
resultados_criterio = {
    'Atraso': [],
    'Valor_Criterio': [],
    'Seed': [],
}
r_otimos = []

print(f"\nIniciando iterações para {num_seeds} sementes...")
for seed in range(1, num_seeds + 1):
    # Gerar sinal e saída
    prbs = prbs_sequence(10, seed)  # Geração do sinal em cada seed
    u_seed = 2 * np.array(prbs.tolist(), dtype=float) - 1
    y_seed = signal.lfilter(b, a, u_seed)

    # Ruído
    Py = np.mean(y ** 2)  # Potência do sinal e desvio padrão
    Pruido = Py / (10 ** (SNR_dB / 10))
    dpadrao = np.sqrt(Pruido)
    gerador_ruido = np.random.default_rng(seed)
    ruido = gerador_ruido.normal(loc=0, scale=dpadrao, size=y_seed.shape)
    y_ruido = y_seed + ruido  # Adicionando o ruído a cada iteração

    # Construir matriz Ay
    Ay = np.zeros((nlin, ncol))
    # Construção da matriz Ay: inverto os termos para que ela fique no formato y(k-1) até y(k-n)
    for k in range(nlin):
        it = p_ini + k
        Ay[k, :] = -y_ruido[it - 1 : it - ncol - 1 : -1]

    # SVD e variável critério usando o r escolhido
    _, S, Vt = np.linalg.svd(Ay, full_matrices=False)  # Método SVD
    Vr = Vt[:r, :].T
    # Fatiamento da matriz Vt do SVD com o fator 'r' do método do artigo

    R = Vr @ Vr.T
    coluna_1 = R[:, 0]
    criterio = 1 / np.abs(coluna_1)  # Cálculo do critério (representa qual atraso específico)

    # Cálculo do 'r' ótimo
    p = len(S)
    erros = np.zeros(p)  # Inicializa a variável erros com o comprimento da matriz de valores singulares
    erros[0] = np.inf
    erros[-1] = np.inf
    erros[1] = np.inf
    erros[-2] = np.inf
    # Define esses valores como np.inf para que não haja falsos positivos no método abaixo

    for q in range(1, p - 1):  # Testa todas as posições possíveis de corte na lista de valores singulares
        grupo_sinal = S[:q]  # Para cada corte divide os valores em dois grupos
        grupo_ruido = S[q:]
        var_sinal = np.sum((grupo_sinal - np.mean(grupo_sinal)) ** 2)  # Variância
        var_ruido = np.sum((grupo_ruido - np.mean(grupo_ruido)) ** 2)
        erros[q] = var_sinal + var_ruido  # Soma as variâncias; o melhor corte é onde essa soma é menor

    r_otimo_seed = np.argmin(erros)  # Guarda o melhor valor para a seed
    r_otimos.append(r_otimo_seed)

    # Armazenar critérios para boxplot
    resultados_criterio['Atraso'].extend(range(1, ncol + 1))
    resultados_criterio['Valor_Criterio'].extend(criterio.tolist())
    resultados_criterio['Seed'].extend([seed] * ncol)

# ------------------------- Análise do SVD -------------------------
frequencias = collections.Counter(r_otimos)  # Mostra quantas vezes cada r apareceu
moda = np.array([frequencias[k] for k in frequencias])

valores, contagens = np.unique(r_otimos, return_counts=True)
moda_porc = contagens / contagens.sum()
tabela = pd.DataFrame({
    'r_unico': valores,
    'moda_%': 100 * moda_porc
})
print('\n Valores possíveis de r')
print(tabela.to_string(index=False))

plt.figure(figsize=(8, 5))


plt.boxplot(r_otimos, vert=True, showfliers=True, patch_artist=True, labels=[nome_sistema])
x_jitter = np.random.normal(1, 0.03, size=len(r_otimos))
plt.plot(x_jitter, r_otimos, 'k.', alpha=0.2)

plt.ylabel('r')
plt.title('Boxplot r')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ------------------------- Identificação Mínimos Quadrados (SysIdentPy) -------------------------
# Ruído
seed = 14
prbs = prbs_sequence(10, seed)  # Geração do sinal em cada seed
u_seed = 2 * np.array(prbs.tolist(), dtype=float) - 1
y_seed = signal.lfilter(b, a, u_seed)

# Ruído
Py = np.mean(y ** 2)  # Potência do sinal e desvio padrão
Pruido = Py / (10 ** (SNR_dB / 10))
dpadrao = np.sqrt(Pruido)
gerador_ruido = np.random.default_rng(seed)
ruido = gerador_ruido.normal(loc=0, scale=dpadrao, size=y_seed.shape)
y_ruido = y + ruido
y_total = y_ruido.reshape(-1, 1)
u_total = u.reshape(-1, 1)

# Divisão priorizada: 50% início para treino, 50% fim para validação
div = int(0.5 * len(y_total))
y_train = y_total[:div]
u_train = u_total[:div]
y_vali = y_total[div:]
u_vali = u_total[div:]

xlags = list(range(1, 35))  # Seleciona atrasos de x de 1 a 34
basis_function = Polynomial(degree=1)
estimator = LeastSquares(unbiased=True)

model = FROLS(
    order_selection=False,
    n_terms=n_terms,
    ylag=ylags,
    xlag=xlags,
    basis_function=basis_function,
    model_type='NARMAX',
    estimator=estimator,
)
# Estima o modelo inicialmente
model.fit(X=u_train, y=y_train)

#print("\n--- Resultados Iniciais do Modelo ---")
D = pd.DataFrame(
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

# Limpeza de regressores nulos
cte = ~np.all(model.final_model == 0, axis=1)  # Remove o termo constante automaticamente
model.final_model = model.final_model[cte, :]
model.theta = model.theta[cte, :]
if model.err is not None and len(model.err) == len(cte):
    model.err = model.err[cte]
model.n_terms = model.final_model.shape[0]

reduced_model = model.final_model.copy()

# Reestimação via SimulateNARMAX
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
    X_test=u_vali,
    y_test=y_vali,
    model_code=reduced_model,
)

#print("\n--- Modelo Final (Após reestimação) ---")
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
print(F.to_string(index=False))
yhat = simulator.predict(X=u_vali, y=y_vali)  # Predição do modelo

rrse = root_relative_squared_error(y_vali, yhat)  # Erros
rmse = root_mean_squared_error(y_vali, yhat)
print("\nMétricas de Erro:")
print(f"Erro RRSE: {rrse:.5f}")
print(f"RMSE: {rmse:.5f}")

# ---------------------- PLOTS ----------------------
# 1. Boxplot do critério
from matplotlib.patches import Patch
df_criterio = pd.DataFrame(resultados_criterio)

# Adicione esta linha para filtrar o lag 1
df_criterio = df_criterio[df_criterio['Atraso'] > 1]

fig, ax = plt.subplots(figsize=(15, 6))

atrasos = sorted(df_criterio['Atraso'].unique())
dados_boxplot = [
    df_criterio.loc[df_criterio['Atraso'] == atraso, 'Valor_Criterio'].values
    for atraso in atrasos
]

# Criação do boxplot
bp = ax.boxplot(dados_boxplot, positions=atrasos, showfliers=False, patch_artist=True)

# Configuração de cores
cor_padrao = 'lightgray'
cor_destaque = '#d62728'

# Iterando sobre cada caixa do boxplot para aplicar o destaque
for patch, atraso in zip(bp['boxes'], atrasos):
    if atraso in ylags:
        patch.set_facecolor(cor_destaque)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
        ax.axvline(x=atraso, color=cor_destaque, linestyle='--', alpha=0.3, zorder=0)
    else:
        patch.set_facecolor(cor_padrao)
        patch.set_edgecolor('gray')
        patch.set_alpha(0.6)

ax.set_yscale('log')
ax.set_title("Distribution of the 'Criterion' Variable (SVD Pre-Identification)")
ax.set_ylabel("Criterion Cₘ")
ax.set_xlabel("Sample index (lag)")

# Criando uma legenda customizada para explicar as cores
legend_elements = [
    Patch(facecolor=cor_padrao, edgecolor='gray', alpha=0.6, label='Discarded lags'),
    Patch(facecolor=cor_destaque, edgecolor='black', label=f'Selected Lags (r={r})')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xticks(atrasos) # Força mostrar todos os números no eixo X
plt.tight_layout()
plt.show()

# 2. Validação do modelo
plt.figure(figsize=(14, 5))
plt.plot(y_vali, label='Experimental Data', color='black')
plt.plot(yhat, label='Estimated Model', color='red', linestyle='--')
plt.ylabel("Amplitude")
plt.xlabel("Samples")
plt.title("Validation of the Estimated Model")
plt.legend()
plt.grid(True)
plt.show()

print("\nResumo das Amplitudes:")
print(f"Amplitude Máx. do Sinal: {np.max(y):.4f}")
print(f"Amplitude Máx. do Ruído: {np.max(ruido):.4f}")
