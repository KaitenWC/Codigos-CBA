# Estratégia Robusta de Amostragem Não-Uniforme Baseada em SVD para Identificação de Sistemas ARX

Este repositório contém os algoritmos e scripts de simulação desenvolvidos para o artigo submetido ao **Congresso Brasileiro de Automática (CBA)**. O projeto implementa uma metodologia estatística robusta para a seleção de atrasos (*lags*) em modelos polinomiais estruturados, utilizando a Decomposição em Valores Singulares (SVD) aplicada à matriz de dados de saída, avaliada sob múltiplas sementes de ruído.

O objetivo principal é otimizar a seleção da estrutura do modelo (termos autoregressivos), garantindo robustez estatística mesmo na presença de ruído com relação sinal-ruído (SNR) de 20 dB.

## 🛠️ Funcionalidades do Código

* **Geração de Sinais PRBS:** Utiliza sequências binárias pseudo-aleatórias (`prbs_sequence`) como sinal de excitação persistente.
* **Análise Estatística Multissemente:** Simulação robusta avaliando $100.000$ sementes diferentes para determinar a ordem de corte ótima ($r$) do SVD por meio da minimização da soma das variâncias do sinal e do ruído.
* **Identificação Algorítmica:** Pré-identificação de estruturas de modelos polinomiais baseada na variável critério $C_m$ extraída das matrizes ortogonais do SVD.
* **Estimação via SysIdentPy:** Estimação de parâmetros via Mínimos Quadrados Ortogonais Iterativos (FROLS) com remoção automática do termo constante (bias) para validação do modelo ARX.

## 📈 Sistemas Modelados

O script permite alternar e validar a metodologia em dois cenários distintos (ajustáveis na variável `escolha` do código):
1.  **Conversor CC-CC Buck:** Modelo de planta eletrônica de potência.
2.  **Sistema Linear (Benchmark):** Sistema dinâmico linear utilizado como benchmark no artigo.

---
