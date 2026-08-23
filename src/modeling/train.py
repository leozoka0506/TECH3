"""
Pipeline de Machine Learning — Tech Challenge Fase 3
Treina e valida modelos de classificacao para prever o status de alfabetizacao.

Modelos:
  - Regressao Logistica (baseline)
  - Random Forest
  - XGBoost

Saida:
  - reports/model_results.txt  : metricas detalhadas
  - src/modeling/best_model.pkl: melhor modelo serializado
"""

import os
import sys
import io
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Fix encoding Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, accuracy_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from src.preprocessing.data_loader import load_analytical_base, get_client
from src.preprocessing.feature_engineering import (
    build_features, get_feature_columns, encode_target
)

# ── Configs ───────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5
IMAGES_DIR    = os.path.join(ROOT, "images")
REPORTS_DIR   = os.path.join(ROOT, "reports")
MODELS_DIR    = os.path.join(ROOT, "src", "modeling")
os.makedirs(IMAGES_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

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
# 1. CARGA E PREPARACAO
# ══════════════════════════════════════════════════════════════════════════════
def load_and_prepare():
    log("=" * 70)
    log("1. CARGA E PREPARACAO DOS DADOS")
    log("=" * 70)

    client = get_client()
    df = load_analytical_base(client=client)
    df = build_features(df)

    cols      = get_feature_columns()
    num_cols  = cols["numericas"]
    cat_cols  = cols["categoricas"]
    target    = cols["target"]
    drop_cols = [c for c in cols["drop"] if c in df.columns]

    # Preserva 'ano' antes de dropar (usado no split temporal)
    df_ano = df["ano"].copy() if "ano" in df.columns else None

    df_model = df.drop(columns=drop_cols)

    # Codifica target
    y_raw = df_model[target].copy()
    y, mapping = encode_target(y_raw)

    valid_mask = y.notna()
    df_model = df_model[valid_mask].reset_index(drop=True)
    y        = y[valid_mask].reset_index(drop=True)
    if df_ano is not None:
        df_ano = df_ano[valid_mask].reset_index(drop=True)

    existing_num = [c for c in num_cols if c in df_model.columns]
    existing_cat = [c for c in cat_cols if c in df_model.columns]
    X = df_model[existing_num + existing_cat]

    log(f"\n  Shape final: X={X.shape}, y={y.shape}")
    log(f"  Colunas numericas ({len(existing_num)}): {existing_num}")
    log(f"  Colunas categoricas ({len(existing_cat)}): {existing_cat}")
    log(f"\n  Distribuicao do target:")
    inv_mapping = {v: k for k, v in mapping.items()}
    for cls_id, cnt in y.value_counts().sort_index().items():
        log(f"    {cls_id} ({inv_mapping.get(cls_id, '?')}): {cnt} ({cnt/len(y)*100:.1f}%)")

    return X, y, df_ano, existing_num, existing_cat, mapping


# ══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT COM ESTRATIFICACAO
# ══════════════════════════════════════════════════════════════════════════════
def split_data(X, y, df_ano=None):
    log("\n" + "=" * 70)
    log("2. SPLIT TEMPORAL (treino=2023, teste=2024)")
    log("=" * 70)
    log("   Evita data leakage temporal: modelo aprende no passado,")
    log("   e prediz o futuro — cenario realista para politica publica.")

    if df_ano is not None and df_ano.nunique() >= 2:
        anos = sorted(df_ano.unique())
        log(f"\n  Anos: treino={anos[-2]}, teste={anos[-1]}")
        train_mask = (df_ano == anos[-2]).values
        test_mask  = (df_ano == anos[-1]).values
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
    else:
        log("  [AVISO] Apenas 1 ano — usando split aleatorio estratificado 80/20")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y
        )

    log(f"\n  Treino: {X_train.shape[0]:,} amostras")
    log(f"  Teste:  {X_test.shape[0]:,} amostras")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONSTRUCAO DO PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(num_cols, cat_cols, classifier):
    """Monta pipeline de preprocessamento + classificador sem data leakage."""

    # Preprocessamento numerico: imputa mediana + padroniza
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Preprocessamento categorico: imputa moda + encoding ordinal
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ], remainder="drop")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])

    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
# 4. TREINAMENTO E VALIDACAO CRUZADA
# ══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    log(f"\n{'─'*60}")
    log(f"  Modelo: {name}")
    log(f"{'─'*60}")

    # Validacao cruzada estratificada
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=["f1_weighted", "accuracy"],
        return_train_score=True,
        n_jobs=-1
    )

    log(f"\n  Cross-validation ({CV_FOLDS} folds):")
    log(f"    F1 (weighted) val  : {cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}")
    log(f"    F1 (weighted) train: {cv_results['train_f1_weighted'].mean():.4f} (overfitting check)")
    log(f"    Accuracy val       : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")

    # Treina no conjunto completo de treino
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metricas no teste
    f1     = f1_score(y_test, y_pred, average="weighted")
    acc    = accuracy_score(y_test, y_pred)

    log(f"\n  Metricas no conjunto de teste:")
    log(f"    F1 weighted : {f1:.4f}")
    log(f"    Accuracy    : {acc:.4f}")
    log(f"\n  Classification Report:")
    log(classification_report(y_test, y_pred))

    return pipeline, {
        "name":        name,
        "pipeline":    pipeline,
        "cv_f1_mean":  cv_results["test_f1_weighted"].mean(),
        "cv_f1_std":   cv_results["test_f1_weighted"].std(),
        "test_f1":     f1,
        "test_acc":    acc,
        "y_pred":      y_pred,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. MATRIZ DE CONFUSAO
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(results_list, y_test, mapping):
    inv_mapping = {v: k for k, v in mapping.items()}
    class_names = [inv_mapping.get(i, str(i)) for i in sorted(mapping.values())]

    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Matrizes de Confusao — Conjunto de Teste",
                 fontsize=14, fontweight="bold")

    for ax, res in zip(axes, results_list):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, cbar=True)
        ax.set_title(f"{res['name']}\nF1={res['test_f1']:.3f}", fontweight="bold")
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    save_fig("10_confusion_matrices")


# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPARATIVO DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_comparison(results_list):
    names    = [r["name"] for r in results_list]
    f1_vals  = [r["test_f1"]  for r in results_list]
    cv_means = [r["cv_f1_mean"] for r in results_list]
    cv_stds  = [r["cv_f1_std"]  for r in results_list]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, cv_means, width, label="CV F1 (media)",
                   color="#4361EE", alpha=0.85, edgecolor="#0F0F1A")
    ax.errorbar(x - width/2, cv_means, yerr=cv_stds,
                fmt="none", color="#FFD700", linewidth=2, capsize=5)
    bars2 = ax.bar(x + width/2, f1_vals, width, label="Teste F1",
                   color="#F72585", alpha=0.85, edgecolor="#0F0F1A")

    ax.set_xlabel("Modelo")
    ax.set_ylabel("F1-Score (weighted)")
    ax.set_title("Comparativo de Modelos — F1-Score", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 1.1)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_fig("11_model_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log("=" * 70)
    log("PIPELINE DE ML — Tech Challenge Fase 3 | Alfabetizacao")
    log("=" * 70)

    # Carga
    X, y, df_ano, num_cols, cat_cols, mapping = load_and_prepare()
    X_train, X_test, y_train, y_test = split_data(X, y, df_ano)

    log("\n" + "=" * 70)
    log("3. TREINAMENTO DOS MODELOS")
    log("=" * 70)

    n_classes = len(mapping)

    # Definicao dos modelos
    models = {
        "Regressao Logistica (baseline)": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, multi_class="multinomial"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    all_results = []
    for name, clf in models.items():
        pipe = build_pipeline(num_cols, cat_cols, clf)
        _, res = train_and_evaluate(name, pipe, X_train, y_train, X_test, y_test)
        all_results.append(res)

    # Melhor modelo por F1 no teste
    best = max(all_results, key=lambda r: r["test_f1"])

    log("\n" + "=" * 70)
    log("4. RESULTADO FINAL")
    log("=" * 70)
    log(f"\n  Melhor modelo : {best['name']}")
    log(f"  F1 weighted   : {best['test_f1']:.4f}")
    log(f"  Accuracy      : {best['test_acc']:.4f}")

    # Graficos
    plot_confusion_matrix(all_results, y_test, mapping)
    plot_model_comparison(all_results)

    # Salva melhor modelo
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump({
        "pipeline":      best["pipeline"],
        "model_name":    best["name"],
        "mapping":       mapping,
        "num_cols":      num_cols,
        "cat_cols":      cat_cols,
        "test_f1":       best["test_f1"],
        "test_accuracy": best["test_acc"],
    }, model_path)
    log(f"\n  [Salvo] Melhor modelo em: {model_path}")

    # Salva relatorio
    report_path = os.path.join(REPORTS_DIR, "model_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"  [Salvo] Relatorio em: {report_path}")

    return best


if __name__ == "__main__":
    best = main()
    print(f"\nTreinamento concluido! Melhor modelo: {best['name']} | F1={best['test_f1']:.4f}")
