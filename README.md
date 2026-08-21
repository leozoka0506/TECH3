# 🎓 Tech Challenge — Fase 3
## Predição e Inteligência Analítica para Alfabetização no Brasil

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![BigQuery](https://img.shields.io/badge/BigQuery-Google_Cloud-blue?logo=google-cloud)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML_Pipeline-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Em_Desenvolvimento-yellow)

---

## 📌 Contexto do Problema

A alfabetização infantil é um dos principais indicadores do desenvolvimento educacional e social do Brasil. O **Indicador Criança Alfabetizada (ICA)** monitora o percentual de crianças que atingem proficiência mínima em Língua Portuguesa ao final do 2º ano do Ensino Fundamental.

Apesar da disponibilidade de dados públicos, gestores educacionais ainda carecem de ferramentas que permitam **antecipar riscos**, identificar municípios vulneráveis e entender quais fatores têm maior impacto nos indicadores de alfabetização.

---

## 🎯 Objetivo Analítico

Desenvolver um **modelo supervisionado de classificação** capaz de prever se um município será considerado **alfabetizado** ou **não alfabetizado**, utilizando variáveis educacionais, territoriais e socioeconômicas provenientes da camada Gold construída na Fase 2.

---

## 🗄️ Base de Dados

Os dados são provenientes do dataset `alfabetizacao_gold` no **BigQuery (GCP)**, construído na Fase 2 do Tech Challenge:

| Tabela | Descrição | Linhas |
|--------|-----------|--------|
| `indicador_por_municipio` | Base principal — indicadores por município/ano | 23.995 |
| `gap_meta_municipio` | Gap em relação à meta 2030 por município | 5.550 |
| `ranking_uf` | Ranking e estatísticas por estado | 50 |
| `evolucao_nacional` | Evolução histórica nacional | 2 |
| `resumo_por_regiao` | Resumo por região geográfica | 10 |

### Fontes externas utilizadas
> *(a preencher conforme enriquecimento da base)*

---

## 🔬 Etapas de Modelagem

1. **Análise Exploratória de Dados (EDA)**
   - Distribuições, correlações, nulos e outliers
   - Formulação de hipóteses analíticas

2. **Feature Engineering**
   - Criação de variáveis derivadas
   - Encoding de variáveis categóricas
   - Tratamento de data leakage

3. **Pipeline de ML (Scikit-learn)**
   - Imputação de valores faltantes
   - Normalização/padronização
   - Treinamento e validação cruzada

4. **Avaliação e Interpretabilidade**
   - Métricas: F1-Score, ROC-AUC, Precision, Recall
   - Feature Importance e SHAP Values

---

## 🤖 Escolha do Algoritmo

| Modelo | Papel |
|--------|-------|
| Regressão Logística | Baseline interpretável |
| Random Forest | Modelo principal — robusto e interpretável |
| XGBoost | Comparativo — alto desempenho |

A escolha final será baseada no **F1-Score** e na **interpretabilidade via SHAP**.

---

## 📊 Métricas de Avaliação

- **F1-Score** (métrica principal — dados desbalanceados)
- **ROC-AUC**
- **Precision / Recall**
- **Matriz de Confusão**

---

## 📁 Estrutura do Projeto

```
tech-challenge-fase3/
│
├── data/               # Dados locais (não versionados)
├── notebooks/          # Notebooks exploratórios
├── src/
│   ├── preprocessing/  # Carga e preparação dos dados
│   ├── modeling/       # Treinamento e pipeline de ML
│   ├── evaluation/     # Métricas e avaliação
│   └── visualization/  # Geração de gráficos
├── reports/            # Relatórios gerados
├── images/             # Visualizações salvas
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install -r requirements.txt
gcloud auth application-default login
```

### Testar conexão com BigQuery
```bash
python test_connection.py
```

### Executar EDA
```bash
python src/preprocessing/eda.py
```

### Treinar modelo
```bash
python src/modeling/train.py
```

---

## 💡 Insights Encontrados

> *(a preencher após a EDA)*

---

## ⚠️ Limitações do Projeto

> *(a preencher ao final)*

---

## 🏛️ Aplicação Prática para Políticas Públicas

> *(a preencher após análise dos resultados)*

---

## 🔮 Possíveis Evoluções Futuras

> *(a preencher ao final)*

---

## 👤 Autor

**Leonardo Wojcik**  
Pós-graduação em Inteligência Analítica e Ciência de Dados — FIAP  
Tech Challenge — Fase 3 | 2026
