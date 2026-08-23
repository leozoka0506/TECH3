"""
Feature Engineering — Tech Challenge Fase 3
Prepara e enriquece a base analitica para o modelo de ML.
"""

import pandas as pd
import numpy as np


# Meta PNE 2030 usada como referencia
META_PNE_2030 = 80.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica feature engineering na base analitica.
    Retorna um DataFrame enriquecido pronto para o pipeline de ML.

    Features criadas:
    - gap_para_meta_2030       : ja calculado no data_loader (80 - taxa)
    - acima_media_uf           : municipio acima da media estadual
    - acima_media_nacional     : municipio acima da media nacional
    - variacao_taxa            : diferenca 2024 - 2023 por municipio (se disponivel)
    - nivel_risco              : categoria ordinal derivada do gap (Alta, Media, Baixa)
    - proficiencia_normalizada : media_portugues normalizada pela media nacional
    """

    df = df.copy()

    # ── 1. Flag: acima da media estadual ─────────────────────────────────────
    media_uf = df.groupby(["sigla_uf", "ano"])["taxa_alfabetizacao"].transform("mean")
    df["acima_media_uf"] = (df["taxa_alfabetizacao"] >= media_uf).astype(int)

    # ── 2. Flag: acima da media nacional ─────────────────────────────────────
    media_nac = df.groupby("ano")["taxa_alfabetizacao"].transform("mean")
    df["acima_media_nacional"] = (df["taxa_alfabetizacao"] >= media_nac).astype(int)

    # ── 3. Variacao temporal por municipio (2024 - 2023) ─────────────────────
    if df["ano"].nunique() > 1:
        pivot = df.pivot_table(
            index="id_municipio",
            columns="ano",
            values="taxa_alfabetizacao",
            aggfunc="mean"
        )
        anos = sorted(pivot.columns.tolist())
        if len(anos) >= 2:
            pivot["variacao_taxa"] = pivot[anos[-1]] - pivot[anos[-2]]
            df = df.merge(
                pivot[["variacao_taxa"]].reset_index(),
                on="id_municipio",
                how="left"
            )
        else:
            df["variacao_taxa"] = 0.0
    else:
        df["variacao_taxa"] = 0.0

    # ── 4. Nivel de risco baseado no gap ─────────────────────────────────────
    def categorize_risk(gap):
        if pd.isna(gap):
            return "Desconhecido"
        if gap <= 0:
            return "Baixo"      # ja atingiu a meta
        elif gap <= 20:
            return "Medio"
        else:
            return "Alto"

    df["nivel_risco"] = df["gap_para_meta_2030"].apply(categorize_risk)

    # ── 5. Proficiencia normalizada ───────────────────────────────────────────
    media_port_nac = df["media_portugues"].mean()
    df["proficiencia_normalizada"] = df["media_portugues"] / media_port_nac

    # ── 6. Regiao codificada ordinalmente (Norte -> 1, Sul -> 5) ─────────────
    regiao_order = {
        "Norte": 1, "Nordeste": 2, "Centro-Oeste": 3,
        "Sudeste": 4, "Sul": 5
    }
    df["regiao_ord"] = df["regiao"].map(regiao_order).fillna(0).astype(int)

    return df


def get_feature_columns() -> dict:
    """
    Retorna os grupos de colunas usados no pipeline de ML.

    NOTA SOBRE DATA LEAKAGE:
    - taxa_alfabetizacao, gap_para_meta_2030 e nivel_risco sao EXCLUIDOS
      das features porque o target (status_alfabetizacao) e derivado
      diretamente delas. Usa-las causaria data leakage trivial (F1 ~ 1.0).
    - A abordagem correta usa apenas preditores EXTERNOS ao calculo do status:
      proficiencia em portugues, media estadual, regiao e variacao temporal.
    - Abordagem de validacao: treino em 2023, teste em 2024 (temporal split).
    """
    return {
        "numericas": [
            "media_portugues",          # proficiencia em lingua portuguesa
            "taxa_alfabetizacao_uf",    # media do estado (nao do municipio)
            "acima_media_uf",           # flag comparativa estadual
            "acima_media_nacional",     # flag comparativa nacional
            "proficiencia_normalizada", # media_portugues / media_nacional
            "regiao_ord",               # codificacao ordinal da regiao
        ],
        "categoricas": [
            "regiao",
            "sigla_uf",
        ],
        "target": "status_alfabetizacao",
        "drop": [
            "id_municipio",
            "nome_municipio",
            "nome_uf",
            "status_meta_municipio",
            "taxa_nulo_flag",
            # Features com leakage — derivadas do target:
            "taxa_alfabetizacao",
            "gap_para_meta_2030",
            "nivel_risco",
        ],
    }


def encode_target(series: pd.Series) -> tuple[pd.Series, dict]:
    """
    Codifica a variavel-alvo categorica em inteiros ordenados.

    Ordem: Critico=0, Atencao=1, Em progresso=2, Meta praticamente atingida=3
    """
    mapping = {
        "Critico":                     0,
        "Atencao":                     1,
        "Em progresso":                2,
        "Meta praticamente atingida":  3,
    }

    # Normaliza acentos para garantir o match
    normalized = series.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
    encoded = normalized.map(mapping)

    # Se sobrou NaN, usa mapeamento direto sem normalizacao
    if encoded.isna().any():
        fallback = series.map(mapping)
        encoded = encoded.combine_first(fallback)

    return encoded.astype(int), mapping


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.preprocessing.data_loader import load_analytical_base

    df = load_analytical_base()
    df_feat = build_features(df)

    cols = get_feature_columns()
    print(f"\nBase apos feature engineering: {df_feat.shape}")
    print(f"\nFeatures numericas: {cols['numericas']}")
    print(f"Features categoricas: {cols['categoricas']}")
    print(f"\nNovas features:")
    new_cols = ["acima_media_uf", "acima_media_nacional", "variacao_taxa",
                "nivel_risco", "proficiencia_normalizada", "regiao_ord"]
    print(df_feat[new_cols].describe(include="all"))

    y, mapping = encode_target(df_feat[cols["target"]])
    print(f"\nDistribuicao do target codificado:\n{y.value_counts().sort_index()}")
    print(f"Mapping: {mapping}")
