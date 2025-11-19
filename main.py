import pandas as pd

laudos_df = pd.read_parquet("./dados-unificados/laudos_rotulados.parquet")
print("Laudos")
print(laudos_df.info())
print(laudos_df.describe())
print(laudos_df.head())

print()
print("-=" * 10)

historico = pd.read_parquet("./dados-unificados/historico_familiar.parquet")
print("Historico")
print(historico.info())
print(historico.describe())
print(historico.head())

print()
print("-=" * 10)

medicamentos_df = pd.read_parquet("./dados-unificados/medicamentos_features.parquet")
print("Medicamentos")
print(medicamentos_df.info())
print(medicamentos_df.describe())
print(medicamentos_df.head())

print()
print("-=" * 10)

df_final = pd.read_parquet("./dados-unificados/dataset_final_modelo.parquet")
print("Final")
print(df_final.info())
print(df_final.describe())
print(df_final.head().to_string())
print(df_final["TEM_DM1"].unique())


