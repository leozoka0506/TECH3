"""
Avaliacao e Interpretabilidade do Modelo
Tech Challenge Fase 3 — Alfabetizacao no Brasil

- Feature Importance (nativa do modelo)
- SHAP Values (interpretabilidade global e local)
- Responde as perguntas de negocio do desafio
"""

import os
import sys
import io
import warnings
import joblib

warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from src.preprocessing.data_loader import load_analytical_base, get_client
from src.preprocessing.feature_engineering import build_features, get_feature_columns, encode_target

RANDOM_STATE = 42
TEST_SIZE    = 0.20
IMAGES_DIR   = os.path.join(ROOT, "images")
REPORTS_DIR  = os.path.join(ROOT, "reports")
MODELS_DIR   = os.path.join(ROOT, "src", "modeling")

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "#0F0F1A",
    "axes.facecolor":   "#1A1A2E",
    "axes.labelcolor":  "#E0E0E0",
    "axes.titlecolor":  "#FFFFFF",
    "axes.edgecolor":   "#2D2D44",
    "grid.color":       "#2D2D44",
    "text.color":       "#E0E0E0",
    "xtick.color":      "#AAAAAA",
    "ytick.color":      "#AAAAAA",
    "legend.facecolor": "#1A1A2E",
    "legend.edgecolor": "#2D2D44",
})

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

def save_fig(name):
    path = os.path.join(IMAGES_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", facecolor=plt.rcParams["figure.facecolor"])
    plt.close()
    log(f"  [Salvo] images/{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARREGA MODELO E DADOS
# ══════════════════════════════════════════════════════════════════════════════
def load_model_and_data():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo nao encontrado em {model_path}. "
            "Execute src/modeling/train.py primeiro."
        )

    artifact = joblib.load(model_path)
    log(f"[INFO] Modelo carregado: {artifact['model_name']}")
    log(f"[INFO] F1 no teste: {artifact['test_f1']:.4f}")

    client = get_client()
    df_raw = load_analytical_base(client=client)      # df completo (com taxa_alfabetizacao)
    df     = build_features(df_raw)

    cols      = get_feature_columns()
    num_cols  = artifact["num_cols"]
    cat_cols  = artifact["cat_cols"]
    mapping   = artifact["mapping"]
    pipeline  = artifact["pipeline"]

    drop_cols = [c for c in cols["drop"] if c in df.columns]
    df_model  = df.drop(columns=drop_cols)

    y_raw = df_model[cols["target"]].copy()
    y, _  = encode_target(y_raw)
    valid_mask = y.notna()
    df_model = df_model[valid_mask].reset_index(drop=True)
    df_full  = df_raw[valid_mask].reset_index(drop=True)   # preserva taxa_alfabetizacao
    y        = y[valid_mask].reset_index(drop=True)
    df_ano   = df["ano"][valid_mask].reset_index(drop=True) if "ano" in df.columns else None

    X = df_model[num_cols + cat_cols]

    # Split temporal igual ao treino
    if df_ano is not None and df_ano.nunique() >= 2:
        anos       = sorted(df_ano.unique())
        test_mask  = (df_ano == anos[-1]).values
        X_test     = X[test_mask]
        y_test     = y[test_mask]
    else:
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        df_full = df_full  # nao filtramos neste caso

    return pipeline, artifact, X, X_test, y_test, num_cols, cat_cols, mapping, df_full


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE IMPORTANCE NATIVA
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(pipeline, num_cols, cat_cols):
    log("\n" + "=" * 70)
    log("2. FEATURE IMPORTANCE (nativa do modelo)")
    log("=" * 70)

    clf = pipeline.named_steps["classifier"]
    all_cols = num_cols + cat_cols

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).mean(axis=0)
    else:
        log("  Modelo nao suporta feature_importances_.")
        return

    feat_df = pd.DataFrame({
        "feature":    all_cols[:len(importances)],
        "importance": importances[:len(all_cols)],
    }).sort_values("importance", ascending=True)

    log("\n  Feature Importances:")
    log(feat_df.sort_values("importance", ascending=False).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#F72585" if v > feat_df["importance"].median() else "#4361EE"
              for v in feat_df["importance"]]
    bars = ax.barh(feat_df["feature"], feat_df["importance"],
                   color=colors, edgecolor="#0F0F1A")
    ax.set_xlabel("Importancia")
    ax.set_title("Feature Importance — Melhor Modelo", fontsize=14, fontweight="bold")
    ax.axvline(feat_df["importance"].median(), color="#FFD700",
               linestyle="--", linewidth=1.5, label="Mediana")
    ax.legend()

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height()/2,
                f"{w:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    save_fig("12_feature_importance")

    return feat_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. SHAP VALUES
# ══════════════════════════════════════════════════════════════════════════════
def compute_shap(pipeline, X_test, num_cols, cat_cols, mapping):
    log("\n" + "=" * 70)
    log("3. SHAP VALUES — Interpretabilidade Global")
    log("=" * 70)

    # Transforma X_test com o preprocessador
    preprocessor = pipeline.named_steps["preprocessor"]
    clf          = pipeline.named_steps["classifier"]

    X_test_transformed = preprocessor.transform(X_test)
    all_cols = num_cols + cat_cols

    # Amostra para SHAP (rapido)
    n_sample = min(500, X_test_transformed.shape[0])
    idx      = np.random.RandomState(RANDOM_STATE).choice(
        X_test_transformed.shape[0], n_sample, replace=False
    )
    X_sample = X_test_transformed[idx]

    log(f"  Calculando SHAP para {n_sample} amostras...")

    # Escolhe explainer de acordo com o tipo de modelo
    clf_name = type(clf).__name__
    shap_values = None

    if "XGB" in clf_name or "Forest" in clf_name or "Tree" in clf_name:
        try:
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample)
            log("  [TreeExplainer OK]")
        except Exception as e:
            log(f"  [AVISO] TreeExplainer falhou ({type(e).__name__}): {e}")
            log("  Usando PermutationExplainer como fallback...")
            try:
                explainer   = shap.PermutationExplainer(clf.predict, X_sample)
                shap_exp    = explainer(X_sample[:100])
                shap_values = shap_exp.values
                log("  [PermutationExplainer OK]")
            except Exception as e2:
                log(f"  [AVISO] PermutationExplainer tambem falhou: {e2}")
    else:
        try:
            explainer   = shap.LinearExplainer(clf, X_sample)
            shap_values = explainer.shap_values(X_sample)
            log("  [LinearExplainer OK]")
        except Exception as e:
            log(f"  [AVISO] LinearExplainer falhou: {e}")

    if shap_values is None:
        log("  Nenhum explainer funcionou — pulando SHAP.")
        return None

    # Para multi-classe, shap_values e lista [n_classes x n_samples x n_features]
    if isinstance(shap_values, list):
        # Media absoluta entre classes para importancia global
        shap_global = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_global = np.abs(shap_values)

    shap_mean = shap_global.mean(axis=0)
    shap_df   = pd.DataFrame({
        "feature": all_cols[:len(shap_mean)],
        "shap_importance": shap_mean[:len(all_cols)],
    }).sort_values("shap_importance", ascending=False)

    log("\n  SHAP Feature Importance (global, media absoluta):")
    log(shap_df.to_string(index=False))

    # ── Plot SHAP Summary Bar ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap_sorted = shap_df.sort_values("shap_importance", ascending=True)
    colors = ["#F72585" if i >= len(shap_sorted) - 3 else "#4361EE"
              for i in range(len(shap_sorted))]
    ax.barh(shap_sorted["feature"], shap_sorted["shap_importance"],
            color=colors, edgecolor="#0F0F1A")
    ax.set_xlabel("SHAP Importance (media |SHAP|)")
    ax.set_title("SHAP Values — Importancia Global das Features",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("13_shap_importance")

    # Beeswarm so funciona com TreeExplainer/LinearExplainer (nao PermutationExplainer)
    if isinstance(shap_values, list) and len(shap_values) > 0 and shap_values[0].shape[0] == X_sample.shape[0]:
        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor("#0F0F1A")
            ax.set_facecolor("#1A1A2E")
            sv_class0 = shap_values[0]
            shap.summary_plot(
                sv_class0,
                X_sample,
                feature_names=all_cols[:X_sample.shape[1]],
                show=False,
                plot_type="dot",
                max_display=10,
            )
            plt.title(f"SHAP Beeswarm — Classe: {list(mapping.keys())[0]}",
                      fontsize=13, fontweight="bold", color="#FFFFFF")
            plt.tight_layout()
            save_fig("14_shap_beeswarm")
        except Exception as e:
            log(f"  [AVISO] Beeswarm nao gerado: {e}")
    else:
        log("  [INFO] Beeswarm pulado (PermutationExplainer retorna formato incompativel)")

    return shap_df


# ══════════════════════════════════════════════════════════════════════════════
# 4. PERGUNTAS DE NEGOCIO
# ══════════════════════════════════════════════════════════════════════════════
def business_questions(df, pipeline, mapping, feat_importance_df, shap_df):
    log("\n" + "=" * 70)
    log("4. RESPOSTAS AS PERGUNTAS DE NEGOCIO")
    log("=" * 70)

    inv_mapping = {v: k for k, v in mapping.items()}

    # Q1: Quais fatores mais impactam a alfabetizacao?
    log("\n  Q1: Quais fatores mais impactam a alfabetizacao?")
    if shap_df is not None:
        top3_shap = shap_df.head(3)["feature"].tolist()
        log(f"    Top 3 (SHAP): {top3_shap}")
    if feat_importance_df is not None:
        top3_imp = feat_importance_df.sort_values("importance", ascending=False).head(3)["feature"].tolist()
        log(f"    Top 3 (Feature Importance): {top3_imp}")

    # Q2: Quais municipios apresentam maior risco educacional?
    log("\n  Q2: Municipios com maior risco educacional (gap > 40pp):")
    if "gap_para_meta_2030" in df.columns and "nome_municipio" in df.columns:
        high_risk = (
            df[df["gap_para_meta_2030"] > 40]
            .sort_values("gap_para_meta_2030", ascending=False)
            [["nome_municipio", "sigla_uf", "taxa_alfabetizacao", "gap_para_meta_2030"]]
            .head(10)
        )
        log(high_risk.to_string(index=False))

    # Q3: Quais regioes possuem padroes semelhantes?
    log("\n  Q3: Media de taxa de alfabetizacao por regiao:")
    if "regiao" in df.columns:
        regiao_stats = (
            df.groupby("regiao")["taxa_alfabetizacao"]
            .agg(["mean", "std", "count"])
            .round(2)
            .sort_values("mean")
        )
        log(regiao_stats.to_string())

    # Q4: Municipios que podem nao atingir a meta 2030
    log("\n  Q4: Municipios criticos — provavelmente nao atingirao a meta 2030:")
    if "status_alfabetizacao" in df.columns:
        criticos = df[df["status_alfabetizacao"] == "Critico"] if "Critico" in df["status_alfabetizacao"].values \
                   else df[df["status_alfabetizacao"].str.contains("tico", na=False)]
        log(f"    Total de municipios criticos: {len(criticos):,}")
        log(f"    Gap medio nos criticos: {criticos['gap_para_meta_2030'].mean():.2f} pp")

    # Q5: Variaveis com maior influencia
    log("\n  Q5: Variaveis com maior influencia no modelo:")
    if shap_df is not None:
        log(shap_df.head(5).to_string(index=False))

    # Grafico das regioes
    if "regiao" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Perguntas de Negocio — Analise Regional", fontsize=14, fontweight="bold")

        regiao_mean = df.groupby("regiao")["taxa_alfabetizacao"].mean().sort_values()
        colors_reg  = ["#F72585" if v < 60 else "#4CC9F0" for v in regiao_mean.values]
        axes[0].barh(regiao_mean.index, regiao_mean.values, color=colors_reg, edgecolor="#0F0F1A")
        axes[0].axvline(80, color="#FFD700", linestyle="--", linewidth=1.5, label="Meta 2030 (80%)")
        axes[0].set_xlabel("Taxa Media de Alfabetizacao (%)")
        axes[0].set_title("Taxa Media por Regiao")
        axes[0].legend()

        if "gap_para_meta_2030" in df.columns:
            regiao_gap = df.groupby("regiao")["gap_para_meta_2030"].mean().sort_values(ascending=False)
            cols_gap   = ["#F72585" if v > 0 else "#4CC9F0" for v in regiao_gap.values]
            axes[1].barh(regiao_gap.index, regiao_gap.values, color=cols_gap, edgecolor="#0F0F1A")
            axes[1].axvline(0, color="#FFD700", linestyle="--", linewidth=1.5)
            axes[1].set_xlabel("Gap Medio para Meta 2030 (pp)")
            axes[1].set_title("Urgencia por Regiao")

        plt.tight_layout()
        save_fig("15_business_questions")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("=" * 70)
    log("AVALIACAO E INTERPRETABILIDADE — Tech Challenge Fase 3")
    log("=" * 70)

    pipeline, artifact, X, X_test, y_test, num_cols, cat_cols, mapping, df = load_model_and_data()

    feat_df  = plot_feature_importance(pipeline, num_cols, cat_cols)
    shap_df  = compute_shap(pipeline, X_test, num_cols, cat_cols, mapping)
    business_questions(df, pipeline, mapping, feat_df, shap_df)

    report_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\n[OK] Relatorio salvo em: {report_path}")
    log("\nAvaliacao concluida! Verifique a pasta images/ para os graficos gerados.")
