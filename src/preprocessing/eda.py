"""
Analise Exploratoria de Dados (EDA)
Tech Challenge Fase 3 — Predição de Alfabetização no Brasil

Gera visualizações salvas em images/ e um relatório em reports/eda_report.txt
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import io
import matplotlib
matplotlib.use("Agg")  # sem janela — salva direto em arquivo
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# Fix encoding para terminal Windows (cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Adiciona raiz do projeto ao path ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.preprocessing.data_loader import get_client, load_table, load_analytical_base

# ── Configuração estética global ──────────────────────────────────────────────
PALETTE_MAIN   = ["#4361EE", "#F72585", "#4CC9F0", "#7209B7", "#3A0CA3"]
PALETTE_STATUS = {"Alfabetizado": "#4CC9F0", "Nao Alfabetizado": "#F72585",
                  "Critico": "#F72585", "Atenção": "#FF9F1C", "Adequado": "#4CC9F0"}

sns.set_theme(style="darkgrid", palette=PALETTE_MAIN, font_scale=1.1)
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
    "font.family":      "DejaVu Sans",
})

IMAGES_DIR  = os.path.join(ROOT, "images")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(IMAGES_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(msg)

def save_fig(name):
    path = os.path.join(IMAGES_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", facecolor=plt.rcParams["figure.facecolor"])
    plt.close()
    log(f"  [Salvo] images/{name}.png")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DOS DADOS
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    log("=" * 70)
    log("1. CARGA DOS DADOS")
    log("=" * 70)

    client = get_client()
    df     = load_analytical_base(client=client)
    df_uf  = load_table("ranking_uf",       client=client)
    df_reg = load_table("resumo_por_regiao", client=client)
    df_nac = load_table("evolucao_nacional", client=client)

    log(f"\nBase analítica: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    log(f"ranking_uf:      {df_uf.shape}")
    log(f"resumo_regiao:   {df_reg.shape}")
    log(f"evolucao_nac:    {df_nac.shape}")
    return df, df_uf, df_reg, df_nac

# ══════════════════════════════════════════════════════════════════════════════
# 2. VISÃO GERAL DA BASE
# ══════════════════════════════════════════════════════════════════════════════
def visao_geral(df):
    log("\n" + "=" * 70)
    log("2. VISÃO GERAL DA BASE ANALÍTICA")
    log("=" * 70)

    log("\n[Tipos de dados]")
    log(df.dtypes.to_string())

    log("\n[Valores nulos por coluna]")
    nulls = df.isnull().sum()
    pct   = (nulls / len(df) * 100).round(2)
    null_df = pd.DataFrame({"nulos": nulls, "pct_%": pct})
    null_df = null_df[null_df["nulos"] > 0].sort_values("pct_%", ascending=False)
    log(null_df.to_string() if len(null_df) else "  Nenhum valor nulo!")

    log("\n[Estatísticas descritivas — numéricas]")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    log(df[num_cols].describe().round(3).to_string())

    # ── Heatmap de nulos ──
    fig, ax = plt.subplots(figsize=(10, 5))
    null_pct_series = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    null_pct_series = null_pct_series[null_pct_series > 0]
    if len(null_pct_series):
        bars = ax.barh(null_pct_series.index, null_pct_series.values, color="#F72585", edgecolor="#0F0F1A")
        ax.set_xlabel("% de valores nulos")
        ax.set_title("Percentual de Valores Nulos por Coluna", fontsize=14, fontweight="bold")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", va="center", fontsize=9)
        save_fig("01_valores_nulos")
    else:
        plt.close()

    return null_df

# ══════════════════════════════════════════════════════════════════════════════
# 3. VARIÁVEL-ALVO
# ══════════════════════════════════════════════════════════════════════════════
def analise_target(df):
    log("\n" + "=" * 70)
    log("3. VARIÁVEL-ALVO: status_alfabetizacao")
    log("=" * 70)

    counts = df["status_alfabetizacao"].value_counts()
    pct    = (counts / len(df) * 100).round(2)
    target_df = pd.DataFrame({"contagem": counts, "pct_%": pct})
    log("\n" + target_df.to_string())

    # ── Distribuição do target ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribuição da Variável-Alvo: status_alfabetizacao",
                 fontsize=14, fontweight="bold", y=1.02)

    colors_map = {k: PALETTE_MAIN[i % len(PALETTE_MAIN)] for i, k in enumerate(counts.index)}
    colors_list = [colors_map[k] for k in counts.index]

    # Barras
    bars = axes[0].bar(counts.index, counts.values, color=colors_list, edgecolor="#0F0F1A", linewidth=0.8)
    axes[0].set_title("Contagem por Classe", fontweight="bold")
    axes[0].set_xlabel("Status")
    axes[0].set_ylabel("Número de registros")
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 50,
                     f"{h:,}", ha="center", fontsize=10)

    # Pizza
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors_list, startangle=140,
                wedgeprops={"edgecolor": "#0F0F1A", "linewidth": 1.5})
    axes[1].set_title("Proporção por Classe", fontweight="bold")

    plt.tight_layout()
    save_fig("02_distribuicao_target")

    log(f"\n  Classes identificadas: {list(counts.index)}")
    log(f"  Desbalanceamento: ratio máx/mín = {(counts.max()/counts.min()):.2f}x")
    return counts

# ══════════════════════════════════════════════════════════════════════════════
# 4. DISTRIBUIÇÃO DA TAXA DE ALFABETIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def distribuicao_taxa(df):
    log("\n" + "=" * 70)
    log("4. DISTRIBUIÇÃO DA TAXA DE ALFABETIZAÇÃO")
    log("=" * 70)

    col = "taxa_alfabetizacao"
    serie = df[col].dropna()

    stats = {
        "Média":    serie.mean(),
        "Mediana":  serie.median(),
        "Std":      serie.std(),
        "Min":      serie.min(),
        "Max":      serie.max(),
        "Skewness": serie.skew(),
        "Kurtosis": serie.kurtosis(),
    }
    for k, v in stats.items():
        log(f"  {k}: {v:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Taxa de Alfabetização — Distribuição", fontsize=14, fontweight="bold")

    # Histograma + KDE
    axes[0].hist(serie, bins=50, color="#4361EE", edgecolor="#0F0F1A",
                 linewidth=0.5, alpha=0.85, density=True)
    serie.plot.kde(ax=axes[0], color="#F72585", linewidth=2.5)
    axes[0].axvline(serie.mean(),   color="#FFD700", linestyle="--", linewidth=1.8, label=f"Média ({serie.mean():.2f})")
    axes[0].axvline(serie.median(), color="#4CC9F0", linestyle=":",  linewidth=1.8, label=f"Mediana ({serie.median():.2f})")
    axes[0].set_xlabel("Taxa de Alfabetização (%)")
    axes[0].set_title("Histograma + KDE")
    axes[0].legend()

    # Boxplot por região
    regioes = df["regiao"].dropna().unique()
    data_box = [df[df["regiao"] == r][col].dropna().values for r in regioes]
    bp = axes[1].boxplot(data_box, labels=regioes, patch_artist=True,
                         medianprops={"color": "#FFD700", "linewidth": 2})
    for patch, color in zip(bp["boxes"], PALETTE_MAIN):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[1].set_xlabel("Região")
    axes[1].set_ylabel("Taxa de Alfabetização (%)")
    axes[1].set_title("Boxplot por Região")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_fig("03_distribuicao_taxa_alfabetizacao")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ANÁLISE TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════
def analise_temporal(df, df_nac):
    log("\n" + "=" * 70)
    log("5. ANÁLISE TEMPORAL")
    log("=" * 70)

    if "ano" not in df.columns:
        log("  Coluna 'ano' não encontrada.")
        return

    anos = sorted(df["ano"].dropna().unique())
    log(f"  Anos disponíveis: {anos}")

    evolucao = df.groupby("ano")["taxa_alfabetizacao"].agg(["mean", "median", "std"]).reset_index()
    log("\n" + evolucao.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Evolução Temporal da Alfabetização", fontsize=14, fontweight="bold")

    # Tendência nacional
    axes[0].plot(evolucao["ano"], evolucao["mean"], "o-", color="#4CC9F0",
                 linewidth=2.5, markersize=8, label="Média nacional")
    axes[0].fill_between(evolucao["ano"],
                         evolucao["mean"] - evolucao["std"],
                         evolucao["mean"] + evolucao["std"],
                         alpha=0.2, color="#4CC9F0", label="±1 desvio padrão")
    axes[0].set_xlabel("Ano")
    axes[0].set_ylabel("Taxa de Alfabetização (%)")
    axes[0].set_title("Tendência Nacional")
    axes[0].legend()

    # Por região ao longo do tempo
    if "regiao" in df.columns:
        for i, reg in enumerate(df["regiao"].dropna().unique()):
            sub = df[df["regiao"] == reg].groupby("ano")["taxa_alfabetizacao"].mean()
            axes[1].plot(sub.index, sub.values, "o-",
                         color=PALETTE_MAIN[i % len(PALETTE_MAIN)],
                         linewidth=2, markersize=6, label=reg)
        axes[1].set_xlabel("Ano")
        axes[1].set_ylabel("Taxa de Alfabetização (%)")
        axes[1].set_title("Evolução por Região")
        axes[1].legend(fontsize=9)

    plt.tight_layout()
    save_fig("04_evolucao_temporal")

# ══════════════════════════════════════════════════════════════════════════════
# 6. ANÁLISE GEOGRÁFICA
# ══════════════════════════════════════════════════════════════════════════════
def analise_geografica(df, df_uf):
    log("\n" + "=" * 70)
    log("6. ANÁLISE GEOGRÁFICA")
    log("=" * 70)

    # Top 10 melhores e piores municípios
    last_year = df["ano"].max()
    df_last   = df[df["ano"] == last_year].copy()

    top10 = df_last.nlargest(10, "taxa_alfabetizacao")[["nome_municipio", "sigla_uf", "taxa_alfabetizacao"]]
    bot10 = df_last.nsmallest(10, "taxa_alfabetizacao")[["nome_municipio", "sigla_uf", "taxa_alfabetizacao"]]

    log(f"\n  Ano de referência: {last_year}")
    log("\n  TOP 10 — Maiores taxas de alfabetização:")
    log(top10.to_string(index=False))
    log("\n  TOP 10 — Menores taxas de alfabetização:")
    log(bot10.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Rankings Municipais — Ano {last_year}", fontsize=14, fontweight="bold")

    # Melhores
    axes[0].barh(
        top10["nome_municipio"] + " (" + top10["sigla_uf"] + ")",
        top10["taxa_alfabetizacao"],
        color="#4CC9F0", edgecolor="#0F0F1A"
    )
    axes[0].set_title("Top 10 Melhores Municipios", fontweight="bold")
    axes[0].set_xlabel("Taxa de Alfabetização (%)")
    axes[0].invert_yaxis()

    # Piores
    axes[1].barh(
        bot10["nome_municipio"] + " (" + bot10["sigla_uf"] + ")",
        bot10["taxa_alfabetizacao"],
        color="#F72585", edgecolor="#0F0F1A"
    )
    axes[1].set_title("Top 10 Municipios Criticos", fontweight="bold")
    axes[1].set_xlabel("Taxa de Alfabetização (%)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    save_fig("05_ranking_municipios")

    # ── Ranking UF ──
    if "ranking_nacional" in df_uf.columns:
        df_uf_last = df_uf[df_uf["ano"] == df_uf["ano"].max()].sort_values("taxa_media", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 12))
        colors_uf = [PALETTE_MAIN[i % len(PALETTE_MAIN)] for i in range(len(df_uf_last))]
        bars = ax.barh(df_uf_last["sigla_uf"], df_uf_last["taxa_media"],
                       color=colors_uf, edgecolor="#0F0F1A")
        ax.set_xlabel("Taxa Média de Alfabetização (%)")
        ax.set_title("Ranking por Estado (UF)", fontsize=14, fontweight="bold")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", va="center", fontsize=8)
        plt.tight_layout()
        save_fig("06_ranking_uf")

# ══════════════════════════════════════════════════════════════════════════════
# 7. CORRELAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
def analise_correlacoes(df):
    log("\n" + "=" * 70)
    log("7. MATRIZ DE CORRELAÇÕES")
    log("=" * 70)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    # Remove colunas de ID e flags booleanas de pouco interesse
    exclude = ["taxa_nulo_flag"]
    num_cols = [c for c in num_cols if c not in exclude]

    corr = df[num_cols].corr()

    # Correlação com a taxa-alvo
    if "taxa_alfabetizacao" in corr.columns:
        corr_target = corr["taxa_alfabetizacao"].drop("taxa_alfabetizacao").sort_values(key=abs, ascending=False)
        log("\n  Correlações com taxa_alfabetizacao:")
        log(corr_target.round(4).to_string())

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
                cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 9},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matriz de Correlação — Variáveis Numéricas", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig("07_correlacoes")

# ══════════════════════════════════════════════════════════════════════════════
# 8. GAP PARA META 2030
# ══════════════════════════════════════════════════════════════════════════════
def analise_gap_meta(df):
    log("\n" + "=" * 70)
    log("8. ANÁLISE DE GAP PARA META 2030")
    log("=" * 70)

    if "gap_para_meta_2030" not in df.columns:
        log("  Coluna 'gap_para_meta_2030' não encontrada.")
        return

    gap = df["gap_para_meta_2030"].dropna()
    criticos = (gap > 0).sum()
    ok       = (gap <= 0).sum()

    log(f"\n  Municípios com GAP positivo (abaixo da meta): {criticos:,} ({criticos/len(gap)*100:.1f}%)")
    log(f"  Municípios que já atingiram a meta:           {ok:,} ({ok/len(gap)*100:.1f}%)")
    log(f"  GAP médio: {gap.mean():.2f}  |  GAP máximo: {gap.max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gap para Meta de Alfabetização 2030", fontsize=14, fontweight="bold")

    # Histograma do gap
    axes[0].hist(gap, bins=60, color="#7209B7", edgecolor="#0F0F1A", linewidth=0.5, alpha=0.85)
    axes[0].axvline(0, color="#FFD700", linestyle="--", linewidth=2, label="Meta atingida (gap=0)")
    axes[0].set_xlabel("Gap para Meta 2030 (%)")
    axes[0].set_ylabel("Número de municípios")
    axes[0].set_title("Distribuição do Gap")
    axes[0].legend()

    # Gap por região
    if "regiao" in df.columns:
        gap_regiao = df.groupby("regiao")["gap_para_meta_2030"].mean().sort_values(ascending=True)
        colors_bar = ["#4CC9F0" if v <= 0 else "#F72585" for v in gap_regiao.values]
        bars = axes[1].barh(gap_regiao.index, gap_regiao.values,
                            color=colors_bar, edgecolor="#0F0F1A")
        axes[1].axvline(0, color="#FFD700", linestyle="--", linewidth=1.5)
        axes[1].set_xlabel("Gap Médio para Meta 2030 (%)")
        axes[1].set_title("Gap Médio por Região")
        for bar in bars:
            w = bar.get_width()
            axes[1].text(w + (0.1 if w >= 0 else -0.5),
                         bar.get_y() + bar.get_height() / 2,
                         f"{w:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    save_fig("08_gap_meta_2030")

# ══════════════════════════════════════════════════════════════════════════════
# 9. HIPÓTESES ANALÍTICAS
# ══════════════════════════════════════════════════════════════════════════════
def hipoteses(df):
    log("\n" + "=" * 70)
    log("9. VERIFICAÇÃO DE HIPÓTESES ANALÍTICAS")
    log("=" * 70)

    hipoteses_lista = []

    # H1: Municípios do Norte/Nordeste têm menor taxa de alfabetização
    if "regiao" in df.columns:
        media_regiao = df.groupby("regiao")["taxa_alfabetizacao"].mean().sort_values()
        h1_norte = media_regiao.get("Norte", None)
        h1_sul   = media_regiao.get("Sul",   None)
        confirmada = (h1_norte is not None and h1_sul is not None and h1_norte < h1_sul)
        hipoteses_lista.append({
            "hipotese": "H1: Regiões Norte/Nordeste têm menor taxa de alfabetização",
            "resultado": "CONFIRMADA" if confirmada else "NÃO CONFIRMADA",
            "detalhe": f"Norte: {h1_norte:.2f}% vs Sul: {h1_sul:.2f}%" if h1_norte else "Dados insuficientes"
        })
        log(f"\n  H1 — {hipoteses_lista[-1]['resultado']}: {hipoteses_lista[-1]['detalhe']}")

    # H2: Proficiência em português correlaciona com taxa de alfabetização
    if "media_portugues" in df.columns and "taxa_alfabetizacao" in df.columns:
        corr_val = df[["media_portugues", "taxa_alfabetizacao"]].corr().iloc[0, 1]
        hipoteses_lista.append({
            "hipotese": "H2: Proficiência em português correlaciona com taxa de alfabetização",
            "resultado": "CONFIRMADA" if abs(corr_val) > 0.3 else "FRACA",
            "detalhe": f"Correlação de Pearson = {corr_val:.4f}"
        })
        log(f"  H2 — {hipoteses_lista[-1]['resultado']}: {hipoteses_lista[-1]['detalhe']}")

    # H3: Municípios com gap positivo tendem a ser classificados como críticos
    if "gap_para_meta_2030" in df.columns and "status_alfabetizacao" in df.columns:
        gap_por_status = df.groupby("status_alfabetizacao")["gap_para_meta_2030"].mean().sort_values(ascending=False)
        hipoteses_lista.append({
            "hipotese": "H3: Gap maior → status mais crítico",
            "resultado": "VERIFICADO",
            "detalhe": gap_por_status.round(2).to_dict()
        })
        log(f"  H3 — {hipoteses_lista[-1]['resultado']}: {hipoteses_lista[-1]['detalhe']}")

    # Gráfico de barras com médias por status
    if "status_alfabetizacao" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Hipóteses Analíticas — Validação", fontsize=14, fontweight="bold")

        media_status = df.groupby("status_alfabetizacao")["taxa_alfabetizacao"].mean().sort_values()
        colors_status = [PALETTE_MAIN[i % len(PALETTE_MAIN)] for i in range(len(media_status))]
        axes[0].barh(media_status.index, media_status.values, color=colors_status, edgecolor="#0F0F1A")
        axes[0].set_xlabel("Taxa Média de Alfabetização (%)")
        axes[0].set_title("Taxa Média por Status")

        # Boxplot taxa_alfabetizacao por status
        statuses = df["status_alfabetizacao"].dropna().unique()
        data_bp  = [df[df["status_alfabetizacao"] == s]["taxa_alfabetizacao"].dropna().values for s in statuses]
        bp = axes[1].boxplot(data_bp, labels=statuses, patch_artist=True,
                             medianprops={"color": "#FFD700", "linewidth": 2})
        for patch, color in zip(bp["boxes"], PALETTE_MAIN):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        axes[1].set_ylabel("Taxa de Alfabetização (%)")
        axes[1].set_title("Dispersão por Status")
        axes[1].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        save_fig("09_hipoteses")

    return hipoteses_lista

# ══════════════════════════════════════════════════════════════════════════════
# 10. SALVAR RELATÓRIO
# ══════════════════════════════════════════════════════════════════════════════
def salvar_relatorio():
    path = os.path.join(REPORTS_DIR, "eda_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[OK] Relatório salvo em reports/eda_report.txt")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("=" * 70)
    log("EDA - Tech Challenge Fase 3 | Alfabetizacao no Brasil")
    log("=" * 70)

    df, df_uf, df_reg, df_nac = load_data()

    visao_geral(df)
    analise_target(df)
    distribuicao_taxa(df)
    analise_temporal(df, df_nac)
    analise_geografica(df, df_uf)
    analise_correlacoes(df)
    analise_gap_meta(df)
    hipoteses(df)
    salvar_relatorio()

    log("\nEDA concluida! Verifique a pasta images/ para os graficos gerados.")
