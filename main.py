import pandas as pd
import os

# ----------------------------------------------------------------------
# 1. CARREGAR DATASETS
# ----------------------------------------------------------------------

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

print()
print("=" * 60)
print("📊 DISTRIBUIÇÃO ORIGINAL DO TARGET (LAUDOS)")
print("=" * 60)

if "TEM_DM1" in laudos_df.columns:
    print(laudos_df["TEM_DM1"].value_counts())
    print("\nProporção:")
    print(laudos_df["TEM_DM1"].value_counts(normalize=True))
else:
    print("[ERRO] TEM_DM1 não encontrado em laudos_rotulados.parquet")


# ----------------------------------------------------------------------
# 2. VERIFICADOR DE INCONSISTÊNCIAS
# ----------------------------------------------------------------------
print()
print("="*60)
print(" 🔎 INICIANDO VERIFICAÇÃO DO DATASET FINAL")
print("="*60)

df = df_final  # só simplificando

# -------------------------------
# 1) Garantir numéricos
# -------------------------------
for col in ["PESO", "ALTURA", "IDADE"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# 2) Duplicados
# -------------------------------
if "CNS" in df.columns:
    duplicados = df["CNS"].duplicated().sum()
    if duplicados == 0:
        print("\n[OK] Nenhum CNS duplicado.")
    else:
        print(f"\n[ALERTA] Existem {duplicados} CNS duplicados!")
else:
    print("\n[INFO] Coluna CNS não encontrada.")

# -------------------------------
# 3) Verificação de NaNs
# -------------------------------
print("\n-- Verificação de NaNs --")
for col in ["PESO", "ALTURA", "SEXO", "RACA_COR"]:
    if col in df.columns:
        faltando = df[col].isna().sum()
        print(f"[OK] {col} está faltando — {faltando} registros")
    else:
        print(f"[INFO] Coluna {col} não encontrada")

# -------------------------------
# 4) Valores absurdos gerais
# -------------------------------
print("\n-- Verificação de valores absurdos gerais --")

if "PESO" in df.columns:
    print(f"[PROBLEMA] Peso <= 0 — {(df['PESO'] <= 0).sum()} registros")
    print(f"[PROBLEMA] Peso > 350 kg — {(df['PESO'] > 350).sum()} registros")
else:
    print("[INFO] Coluna PESO não encontrada.")

if "ALTURA" in df.columns:
    print(f"[PROBLEMA] Altura <= 0 — {(df['ALTURA'] <= 0).sum()} registros")
    print(f"[PROBLEMA] Altura > 250 cm — {(df['ALTURA'] > 250).sum()} registros")
else:
    print("[INFO] Coluna ALTURA não encontrada.")

# -------------------------------
# 5) Valores absurdos por faixa etária
# -------------------------------
print("\n-- Verificação avançada (por faixa etária) --")

if "IDADE" in df.columns:

    RN = df["IDADE"] <= 11
    ADOLESCENTE = (df["IDADE"] >= 12) & (df["IDADE"] <= 17)
    ADULTO = df["IDADE"] >= 18

    # CRIANÇAS / RN
    if RN.sum() > 0:
        print(f"\nCRIANÇAS / RN (0–11 anos): {RN.sum()} registros")
        print(f"[PROBLEMA] Altura > 180 cm — {(df.loc[RN, 'ALTURA'] > 180).sum()} registros")
        print(f"[PROBLEMA] Peso > 120 kg — {(df.loc[RN, 'PESO'] > 120).sum()} registros")

    # ADOLESCENTES
    if ADOLESCENTE.sum() > 0:
        print(f"\nADOLESCENTES (12–17 anos): {ADOLESCENTE.sum()} registros")
        print(f"[PROBLEMA] Altura > 220 cm — {(df.loc[ADOLESCENTE, 'ALTURA'] > 220).sum()} registros")
        print(f"[PROBLEMA] Peso > 200 kg — {(df.loc[ADOLESCENTE, 'PESO'] > 200).sum()} registros")

    # ADULTOS
    if ADULTO.sum() > 0:
        print(f"\nADULTOS (>=18 anos): {ADULTO.sum()} registros")
        print(f"[PROBLEMA] Altura > 250 cm — {(df.loc[ADULTO, 'ALTURA'] > 250).sum()} registros")
        print(f"[PROBLEMA] Peso > 350 kg — {(df.loc[ADULTO, 'PESO'] > 350).sum()} registros")

else:
    print("[INFO] Coluna IDADE não encontrada — não é possível validar faixas etárias.")


print("\n" + "="*60)
print(" ✅ VERIFICAÇÃO FINALIZADA")
print("="*60)


# ----------------------------------------------------------------------
# 6. EXPORTAÇÃO EM CSV DOS DADOS INCONSISTENTES
# ----------------------------------------------------------------------
print("\nSalvando inconsistências em CSV...")

output_dir = "./dados-unificados/relatorios"
os.makedirs(output_dir, exist_ok=True)

absurdos_gerais = []

# Peso <= 0
absurdos_gerais.append(
    df[df["PESO"] <= 0].head(30).assign(PROBLEMA="Peso <= 0")
)

# Peso > 350
absurdos_gerais.append(
    df[df["PESO"] > 400].head(30).assign(PROBLEMA="Peso > 400")
)

# Altura <= 0
absurdos_gerais.append(
    df[df["ALTURA"] <= 0].head(30).assign(PROBLEMA="Altura <= 0")
)

# Altura > 250
absurdos_gerais.append(
    df[df["ALTURA"] > 250].head(30).assign(PROBLEMA="Altura > 250")
)

df_absurdos_gerais = pd.concat(absurdos_gerais, ignore_index=True)
df_absurdos_gerais.to_csv(f"{output_dir}/absurdos_gerais.csv", index=False)
print("[OK] absurdos_gerais.csv salvo.")


# ABSURDOS POR FAIXA ETÁRIA
df_rn_abs = df[(df["IDADE"] <= 11) & ((df["ALTURA"] > 180) | (df["PESO"] > 120))]
df_rn_abs.to_csv(f"{output_dir}/absurdos_criancas_rn.csv", index=False)
print("[OK] absurdos_criancas_rn.csv salvo.")

df_ado_abs = df[(df["IDADE"] >= 12) & (df["IDADE"] <= 17) &
                ((df["ALTURA"] > 220) | (df["PESO"] > 200))]
df_ado_abs.to_csv(f"{output_dir}/absurdos_adolescentes.csv", index=False)
print("[OK] absurdos_adolescentes.csv salvo.")

df_adult_abs = df[(df["IDADE"] >= 18) &
                  ((df["ALTURA"] > 250) | (df["PESO"] > 400))]
df_adult_abs.to_csv(f"{output_dir}/absurdos_adultos.csv", index=False)
print("[OK] absurdos_adultos.csv salvo.")

print()
print("=" * 60)
print("📊 DISTRIBUIÇÃO FINAL DO TARGET (DATASET FINAL)")
print("=" * 60)

if "TEM_DM1" in df_final.columns:
    print(df_final["TEM_DM1"].value_counts())
    print("\nProporção:")
    print(df_final["TEM_DM1"].value_counts(normalize=True))
else:
    print("[ERRO] TEM_DM1 não encontrado no dataset final")

# ----------------------------------------------------------------------
# 7. SALVAR DATASET FINAL CORRETO
# ----------------------------------------------------------------------
print("\n💾 Salvando dataset final para modelagem...")

final_path = "./dados-unificados/dataset_final_modelo.parquet"
df_final.to_parquet(final_path, index=False)

print(f"✅ Dataset final salvo com sucesso em: {final_path}")
print("Shape final:", df_final.shape)

# ----------------------------------------------------------------------
# 8. ONE-HOT ENCODING PARA MODELAGEM
# ----------------------------------------------------------------------
print("\n🧬 Gerando dataset para modelagem (One-Hot Encoding)")

categoricas = ["SEXO", "RACA_COR", "IMC_CLASS"]

df_ohe = pd.get_dummies(
    df_final,
    columns=[c for c in categoricas if c in df_final.columns],
    drop_first=True
)

# -------------------------------
# Validações finais
# -------------------------------
if df_ohe.empty:
    raise ValueError("❌ Dataset OHE vazio")

if "TEM_DM1" not in df_ohe.columns:
    raise ValueError("❌ Coluna TEM_DM1 não encontrada após OHE")

if df_ohe["TEM_DM1"].nunique() < 2:
    raise ValueError("❌ Target inválido após OHE (apenas uma classe)")

# -------------------------------
# Salvando dataset OHE
# -------------------------------
ohe_path = "./dados-unificados/dataset_modelagem_ohe.parquet"
df_ohe.to_parquet(ohe_path, index=False)

print(f"✅ Dataset OHE salvo com sucesso em: {ohe_path}")
print("Shape OHE:", df_ohe.shape)

print("\n🚀 PIPELINE FINALIZADO COM SUCESSO")
