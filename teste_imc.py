import pandas as pd
import config

# Carrega o dataset
df = pd.read_parquet(config.FILE_FEATURES_COMPLETO)

# Calcula correlação
corr = df[['PESO', 'IMC']].corr().iloc[0, 1]

print(f"📉 Correlação entre PESO e IMC: {corr:.4f}")

if corr > 0.80:
    print("✅ PROVA DEFINITIVA: O Peso é altamente correlacionado com o IMC.")
    print("Manter os dois causaria multicolinearidade. A remoção do PESO é obrigatória.")
