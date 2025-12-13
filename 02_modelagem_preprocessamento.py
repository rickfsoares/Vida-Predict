import pandas as pd
import os

# --------------------------------------------------
# 1. CONFIGURAÇÃO
# --------------------------------------------------
PASTA_DADOS = "dados-unificados"
ARQUIVO_ENTRADA = os.path.join(PASTA_DADOS, "dataset_final_modelo.parquet")
ARQUIVO_SAIDA = os.path.join(PASTA_DADOS, "dataset_modelagem_ohe.parquet")

print("\n--- INICIANDO PRÉ-PROCESSAMENTO PARA MODELAGEM ---")

# --------------------------------------------------
# 2. CARREGAR DATASET FINAL LIMPO
# --------------------------------------------------
df = pd.read_parquet(ARQUIVO_ENTRADA)
print(f"Dataset carregado: {df.shape}")

# --------------------------------------------------
# 3. GARANTIR TIPOS CATEGÓRICOS
# --------------------------------------------------
df['RACA_COR'] = df['RACA_COR'].astype(str)
df['IMC_CLASS'] = df['IMC_CLASS'].astype(str)

# --------------------------------------------------
# 4. ONE-HOT ENCODING (RAÇA + IMC_CLASS)
# --------------------------------------------------
df = pd.get_dummies(
    df,
    columns=['RACA_COR', 'IMC_CLASS'],
    prefix=['RACA_COR', 'IMC_CLASS'],
    drop_first=False
)

print("One-Hot Encoding aplicado em RACA_COR e IMC_CLASS")

# --------------------------------------------------
# 5. SALVAR DATASET PARA MODELAGEM
# --------------------------------------------------
df.to_parquet(ARQUIVO_SAIDA)

print(f"Dataset final para modelagem salvo em: {ARQUIVO_SAIDA}")
print(f"Shape final: {df.shape}")
print("--- PROCESSO CONCLUÍDO ---")
