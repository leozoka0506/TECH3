# Tech Challenge Fase 3 — Análise de Alfabetização Municipal no Brasil

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![BigQuery](https://img.shields.io/badge/GCP-BigQuery-orange?logo=googlecloud)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Concluido-4ADE80)

> **IAST (Instituto de Alfabetização, Saúde e Trabalho)**  
> Previsão do status de alfabetização de municípios brasileiros com Machine Learning — Tech Challenge Pós-Graduação FIAP, Fase 3.

---

## 🎯 Objetivo

Desenvolver um **pipeline completo de Data Science** — da engenharia de dados ao modelo de ML em produção — para classificar municípios brasileiros em quatro categorias de risco de alfabetização:

| Status | Descrição |
|---|---|
| 🔴 **Crítico** | Taxa de alfabetização muito baixa, alto risco de não atingir a meta 2030 |
| 🟠 **Atenção** | Situação preocupante, requer intervenção |
| 🔵 **Em progresso** | Avanço positivo, mas ainda abaixo da meta |
| 🟢 **Meta praticamente atingida** | Município próximo ou acima de 80% (PNE 2030) |

---

## 📊 Resultados Principais

### 🏆 Melhor Modelo: XGBoost (split temporal 2023 → 2024)

| Métrica | Valor |
|---|---|
| **F1-Score (weighted)** | **0.859** |
| **Accuracy** | **0.863** |
| Cross-Validation F1 | 0.918 ± 0.002 |

### 📈 Comparativo de Modelos

| Modelo | CV F1 | F1 Teste | Accuracy |
|---|---|---|---|
| Regressão Logística (baseline) | 0.896 ± 0.004 | 0.856 | 0.856 |
| Random Forest | 0.904 ± 0.005 | 0.850 | 0.854 |
| **XGBoost ⭐** | **0.918 ± 0.002** | **0.859** | **0.863** |

### 🔍 SHAP Values — Feature Importance

| Rank | Feature | Importância (SHAP) |
|---|---|---|
| 1 | `acima_media_nacional` | 0.424 |
| 2 | `media_portugues` | 0.206 |
| 3 | `acima_media_uf` | 0.078 |
| 4 | `proficiencia_normalizada` | 0.070 |
| 5 | `taxa_alfabetizacao_uf` | 0.053 |

> **Insight:** a posição relativa do município frente à média nacional é o preditor dominante — 2× mais importante que a proficiência bruta em português.

---

## 🗂️ Estrutura do Projeto

```
tech-challenge-fase3/
├── data/                       # Dados locais (não versionados)
├── images/                     # 21 gráficos gerados
│   ├── 01-09_eda/             # EDA
│   ├── 10-15_modeling/        # Modelagem e SHAP
│   └── 16-21_adicionais/      # Visualizações extras + dashboard
├── notebooks/                  # (reservado para exploração)
├── reports/
│   ├── eda_report.txt          # Relatório completo da EDA
│   ├── model_results.txt       # Métricas dos 3 modelos
│   └── evaluation_report.txt   # SHAP + Perguntas de negócio
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py      # Integração BigQuery + SQL com gap sintético
│   │   ├── eda.py              # Análise Exploratória (9 seções)
│   │   └── feature_engineering.py  # 6 features + tratamento de leakage
│   ├── modeling/
│   │   └── train.py            # Pipeline Scikit-learn (3 modelos)
│   ├── evaluation/
│   │   └── evaluate.py         # SHAP Values + perguntas de negócio
│   └── visualization/
│       └── visualizacoes_adicionais.py  # 6 gráficos extras + dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install -r requirements.txt
gcloud auth application-default login
```

### Passo a passo

```bash
# 1. EDA completa (gera images/01-09 e reports/eda_report.txt)
python src/preprocessing/eda.py

# 2. Treinamento dos modelos (gera images/10-11 e best_model.pkl)
python src/modeling/train.py

# 3. Avaliação + SHAP (gera images/12-15 e reports/evaluation_report.txt)
python src/evaluation/evaluate.py

# 4. Visualizações adicionais (gera images/16-21)
python src/visualization/visualizacoes_adicionais.py
```

---

## 🔬 Decisões Técnicas

### Variável-alvo
`status_alfabetizacao` — 4 classes derivadas do gap para a meta PNE 2030.

### Prevenção de Data Leakage
As features `taxa_alfabetizacao`, `gap_para_meta_2030` e `nivel_risco` foram **excluídas** do modelo por serem derivadas diretamente da variável-alvo. O uso ingênuo dessas features resulta em F1=1.0 (leakage trivial).

### Split Temporal (não aleatório)
- **Treino:** dados de 2023 (**5.514 municípios**)
- **Teste:** dados de 2024 (**5.516 municípios**)

Simula o cenário real: *"com os dados de hoje, consigo prever o status educacional do município no próximo ciclo?"*

### Meta PNE 2030
A coluna `gap_para_meta_2030` nas tabelas do BigQuery está 100% nula (não populada na Fase 2). Foi calculada sinteticamente como `80% - taxa_alfabetizacao`, alinhada ao **Plano Nacional de Educação**.

---

## 📍 Principais Insights da EDA

| Hipótese | Resultado |
|---|---|
| **H1** — Desigualdade regional existe | ✅ Norte: 48,65% vs Sul: 68,43% (brecha de 19,8 pp) |
| **H2** — Proficiência em português prediz taxa | ✅ Pearson = 0.926 |
| **H3** — Gap cresce com piora do status | ✅ Crítico +34,9 pp abaixo da meta |

- **95,4%** dos municípios estão abaixo da meta de 80% para 2030
- **5.165 municípios** classificados como "Crítico" (46,8% da base)
- Desbalanceamento de **10,1×** entre a maior e menor classe (5.165 vs 510)
- Cobertura: **5.514–5.516 municípios/ano** dos **5.570 brasileiros** (99%)

---

## ⚙️ Infraestrutura

- **Fonte dos dados:** Google BigQuery (`tech2-499614.alfabetizacao_gold`)
- **Tabelas:** `indicador_por_municipio`, `gap_meta_municipio`, `ranking_uf`, `resumo_por_regiao`, `evolucao_nacional`
- **Autenticação:** `gcloud auth application-default login`

---

## 🏛️ Aplicação para Políticas Públicas

Este projeto foi construído para apoiar **gestores públicos e formuladores de políticas educacionais**. Os resultados podem ser aplicados de forma prática:

| Uso | Como aplicar |
|---|---|
| **Priorização de intervenções** | Identificar municípios Críticos (gap > 40pp) para alocação emergencial de recursos |
| **Monitoramento preditivo** | Usar o modelo anualmente para antecipar municípios em deterioração antes dos dados oficiais |
| **Foco na causa raiz** | Investir em ensino de língua portuguesa — preditor com SHAP=0.206, maior alavanca comprovada |
| **Política regional** | Tratar Norte (48,65%) e Nordeste (54,82%) com programas específicos — não genéricos |
| **Metas realistas** | Municípios Críticos precisam avançar 3 pp/ano até 2030 — 4× o ritmo histórico; exige intervenção estrutural |

---

## ⚠️ Limitações do Projeto

| Limitação | Impacto | Sugestão de Mitigação |
|---|---|---|
| **Apenas 2 anos de dados** (2023–2024) | Split temporal tem pouco histórico; tendências de longo prazo não capturadas | Incorporar anos anteriores quando disponíveis |
| **Ausência de dados socioeconômicos** (IBGE, PNAD, IDH) | Modelo usa apenas indicadores educacionais — fatores estruturais não observados | Enriquecer base com Atlas do Desenvolvimento Humano e Censo Escolar |
| **gap_para_meta_2030 sintético** | A coluna original estava 100% nula no BigQuery; gap foi calculado como `80% - taxa_alfabetizacao` | Aguardar dados oficiais da Fase 2 corrigidos |
| **Desbalanceamento de classes** (10:1) | Classe minoritária ("Meta atingida") com menor recall mesmo com `class_weight=balanced` | SMOTE ou técnicas de oversampling específicas |
| **Variável-alvo dependente de thresholds** | `status_alfabetizacao` é definido por intervalos — mudança nos limites altera completamente as classes | Considerar regressão (taxa contínua) como objetivo alternativo |
| **Sem dados de investimento público** | Não captura efeito de políticas educacionais específicas em andamento | Integrar dados do FUNDEB por município |

---

## 👤 Autores

Projeto desenvolvido como Tech Challenge da Pós-Graduação em Data Analytics — FIAP.
