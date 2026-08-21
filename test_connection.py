from google.cloud import bigquery

client = bigquery.Client(project="tech2-499614")
tables = ["indicador_por_municipio", "gap_meta_municipio", "evolucao_nacional", "ranking_uf", "resumo_por_regiao"]

print("=== Conexao BigQuery ===")
for t in tables:
    q = "SELECT COUNT(*) as n FROM `tech2-499614.alfabetizacao_gold." + t + "`"
    result = list(client.query(q).result())
    print(f"  {t}: {result[0].n} linhas")

print("\nConexao OK!")
