import pandas as pd

#df = pd.read_parquet("./dados-datasus-processados/PASP0001.parquet/")

df = pd.read_parquet("./dados-datasus-processados/AMSP2201.parquet")
print("Info APAC de Medicamentos")
print(df.info())
print("Describe")
print(df.describe())

df2 = pd.read_parquet("./dados-datasus-processados/ADSP2201.parquet")
print("Info APAC de Laudos Diversos")
print(df2.info())
print("describe")
print(df2.describe())
print()

df3 = pd.read_parquet("./dados-datasus-processados/BISP2201.parquet")
print("Info Boletim de Produção Ambulatorial individualizado")
print(df3.info())
print("describe")
print(df3.describe())

#df = pd.read_parquet("./dados-unificados/medicamentos_features.parquet")
#print("Info")
#print(df.info())
#print("DESCRIBE")
#print(df.describe())
#print()
#print(df.head())
