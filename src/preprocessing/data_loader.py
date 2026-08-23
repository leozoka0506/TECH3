"""
Módulo de carregamento de dados do BigQuery — Camada Gold (Fase 2)
Projeto: Tech Challenge Fase 3 — Predição de Alfabetização
"""

from google.cloud import bigquery
import pandas as pd
import os

PROJECT_ID = "tech2-499614"
DATASET = "alfabetizacao_gold"

# Tabelas disponíveis na camada Gold
TABLES = {
    "indicador_por_municipio": f"{PROJECT_ID}.{DATASET}.indicador_por_municipio",
    "gap_meta_municipio":      f"{PROJECT_ID}.{DATASET}.gap_meta_municipio",
    "evolucao_nacional":       f"{PROJECT_ID}.{DATASET}.evolucao_nacional",
    "ranking_uf":              f"{PROJECT_ID}.{DATASET}.ranking_uf",
    "resumo_por_regiao":       f"{PROJECT_ID}.{DATASET}.resumo_por_regiao",
}


def get_client() -> bigquery.Client:
    """Retorna cliente autenticado do BigQuery."""
    return bigquery.Client(project=PROJECT_ID)


def load_table(table_name: str, client: bigquery.Client = None) -> pd.DataFrame:
    """
    Carrega uma tabela do dataset Gold como DataFrame.

    Args:
        table_name: Nome da tabela (chave de TABLES).
        client: Cliente BigQuery (opcional; cria um novo se None).

    Returns:
        DataFrame com os dados da tabela.
    """
    if table_name not in TABLES:
        raise ValueError(f"Tabela '{table_name}' não encontrada. Disponíveis: {list(TABLES.keys())}")

    if client is None:
        client = get_client()

    full_table = TABLES[table_name]
    print(f"[INFO] Carregando tabela: {full_table} ...")
    query = f"SELECT * FROM `{full_table}`"
    df = client.query(query).to_dataframe()
    print(f"[INFO] {len(df):,} linhas carregadas de '{table_name}'")
    return df


def load_all_tables(client: bigquery.Client = None) -> dict[str, pd.DataFrame]:
    """
    Carrega todas as tabelas do dataset Gold.

    Returns:
        Dicionário {nome_tabela: DataFrame}
    """
    if client is None:
        client = get_client()

    dataframes = {}
    for name in TABLES:
        dataframes[name] = load_table(name, client=client)
    return dataframes


def load_analytical_base(client: bigquery.Client = None) -> pd.DataFrame:
    """
    Constrói a base analítica principal para o modelo de ML,
    unindo indicador_por_municipio com gap_meta_municipio.

    Esta é a tabela base para a EDA e modelagem.

    Returns:
        DataFrame enriquecido e pronto para análise.
    """
    if client is None:
        client = get_client()

    # NOTA: gap_para_meta_2030 e meta_2030 estão NULL na tabela gap_meta_municipio
    # (não foram populados na Fase 2). O gap é calculado sinteticamente abaixo
    # com base em uma meta de referência de 80% (alinhada ao PNE 2030).
    query = """
        WITH base AS (
            SELECT
                i.id_municipio,
                i.nome_municipio,
                i.sigla_uf,
                i.nome_uf,
                i.regiao,
                i.ano,
                i.taxa_alfabetizacao,
                i.media_portugues,
                i.taxa_alfabetizacao_uf,
                i.status_alfabetizacao,
                i.taxa_nulo_flag,
                g.status_alfabetizacao AS status_meta_municipio
            FROM
                `tech2-499614.alfabetizacao_gold.indicador_por_municipio` i
            LEFT JOIN
                `tech2-499614.alfabetizacao_gold.gap_meta_municipio` g
            ON
                i.id_municipio = g.id_municipio
        )
        -- Agrega por municipio/ano (elimina duplicatas por rede)
        -- Calcula gap sintetico usando meta PNE 2030 = 80%
        SELECT
            id_municipio,
            nome_municipio,
            sigla_uf,
            nome_uf,
            regiao,
            ano,
            AVG(taxa_alfabetizacao)         AS taxa_alfabetizacao,
            AVG(media_portugues)            AS media_portugues,
            AVG(taxa_alfabetizacao_uf)      AS taxa_alfabetizacao_uf,
            80.0 - AVG(taxa_alfabetizacao)  AS gap_para_meta_2030,
            MAX(status_alfabetizacao)       AS status_alfabetizacao,
            MAX(status_meta_municipio)      AS status_meta_municipio,
            LOGICAL_OR(taxa_nulo_flag)      AS taxa_nulo_flag
        FROM base
        GROUP BY
            id_municipio, nome_municipio, sigla_uf, nome_uf, regiao, ano
    """

    print("[INFO] Construindo base analítica (JOIN indicador + gap_meta)...")
    df = client.query(query).to_dataframe()
    print(f"[INFO] Base analítica: {len(df):,} linhas x {len(df.columns)} colunas")
    return df


if __name__ == "__main__":
    # Teste rápido de conexão e carga
    client = get_client()
    print("=" * 60)
    print("Resumo das tabelas disponíveis:")
    print("=" * 60)
    for name in TABLES:
        df = load_table(name, client=client)
        print(f"  {name}: {df.shape[0]:,} linhas, {df.shape[1]} colunas")
        print(f"    Colunas: {list(df.columns)}\n")
