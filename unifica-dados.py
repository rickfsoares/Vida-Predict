import pandas as pd
import os
import glob
import pyarrow.parquet as pq
import numpy as np


# --- 1. DEFINIÇÕES ---
PASTA_DADOS_BRUTOS_PARQUET = 'dados-datasus-processados'
PASTA_DADOS_UNIFICADOS = 'dados-unificados'
os.makedirs(PASTA_DADOS_UNIFICADOS, exist_ok=True)

CID_ALVO_DM1 = 'E10' 
CID_HISTORICO_FAMILIAR = 'Z833'
ARQUIVO_FINAL_MODELO = os.path.join(PASTA_DADOS_UNIFICADOS, 'dataset_final_modelo.parquet')

# --- 2. LISTAS PARA COLETAR OS "CHUNKS" ---
lista_laudos_rotulados = []
lista_medicamentos = []
lista_historico_filtrados = []

print("--- INICIANDO UNIFICAÇÃO DE CHUNKS ---")

# --- 3. COLETANDO TUDO (RÓTULOS, DEMOGRÁFICOS e FEATURES) A PARTIR DE AMSP ---
colunas_necessarias_amsp = ['AP_CNSPCN', 'AP_CIDPRI', 'AP_NUIDADE', 'AP_SEXO', 'AP_RACACOR', 'AM_PESO', 'AM_ALTURA']

padrao_am = os.path.join(PASTA_DADOS_BRUTOS_PARQUET, 'AMSP*.parquet') 
arquivos_am = glob.glob(padrao_am)
print(f"\nProcessando {len(arquivos_am)} arquivos de Medicamentos (AM) para Rótulos e Features...")

for parquet_file in arquivos_am:
    try:
        # Lê todas as colunas necessárias de uma vez
        df_chunk = pd.read_parquet(parquet_file, columns=colunas_necessarias_amsp)
        
        # 1. RÓTULOS & DEMOGRÁFICOS (Para lista_laudos_rotulados)
        df_rotulos = df_chunk[['AP_CNSPCN', 'AP_CIDPRI', 'AP_NUIDADE', 'AP_SEXO', 'AP_RACACOR']].copy()
        df_rotulos['TEM_DM1'] = df_rotulos['AP_CIDPRI'].str.startswith(CID_ALVO_DM1, na=False).astype(int)
        lista_laudos_rotulados.append(df_rotulos)

        # 2. FEATURES (Para lista_medicamentos)
        df_features = df_chunk[['AP_CNSPCN', 'AM_PESO', 'AM_ALTURA']].copy()
        lista_medicamentos.append(df_features)

    except Exception as e:
        print(f"  -> Erro ao processar {parquet_file}: {e}")

# --- 4. LAUDOS (ADSP) ---
padrao_ad = os.path.join(PASTA_DADOS_BRUTOS_PARQUET, 'ADSP*.parquet') 
arquivos_ad = glob.glob(padrao_ad)
print(f"\nIgnorando {len(arquivos_ad)} arquivos de Laudos (AD).")


# --- 5. HISTÓRICO (BI) - USANDO LEITURA FRAGMENTADA SEGURA ---
colunas_historico_necessarias = ['CNS_PAC', 'CIDPRI']
padrao_bi = os.path.join(PASTA_DADOS_BRUTOS_PARQUET, 'BISP*.parquet')
arquivos_bi = glob.glob(padrao_bi)

print(f"\nProcessando {len(arquivos_bi)} diretórios de BPA-I (BI)...")

for parquet_directory in arquivos_bi:
    try:
        print(f"  Processando {os.path.basename(parquet_directory)} em fragmentos (modo seguro)...")
        
        dataset = pq.ParquetDataset(parquet_directory)
        
        for fragment in dataset.fragments:
            df_chunk = fragment.to_table(columns=colunas_historico_necessarias).to_pandas()
            
            filtro_hist = df_chunk['CIDPRI'].str.strip().str.upper() == CID_HISTORICO_FAMILIAR
            df_filtrado = df_chunk[filtro_hist]
            
            if not df_filtrado.empty:
                lista_historico_filtrados.append(df_filtrado[['CNS_PAC']]) 
                
        print(f"  -> {os.path.basename(parquet_directory)} concluído.")

    except Exception as e:
        print(f"  -> Erro ao processar {parquet_directory}: {e}")


# --- 6. UNIFICAR E SALVAR ARQUIVOS INTERMEDIÁRIOS ---
print("\n--- Unificando e Salvando Arquivos Intermediários ---")

# --- Laudos (Rotulados) ---
if lista_laudos_rotulados:
    df_laudos = pd.concat(lista_laudos_rotulados, ignore_index=True)
    df_laudos = df_laudos.rename(columns={'AP_CNSPCN': 'CNS', 'AP_CIDPRI': 'CID_DIAGNOSTICO',
                                          'AP_NUIDADE': 'IDADE', 'AP_SEXO': 'SEXO', 'AP_RACACOR': 'RACA_COR'})
                                          
    df_laudos = df_laudos.drop_duplicates(subset=['CNS'], keep='first')
    df_laudos.to_parquet(os.path.join(PASTA_DADOS_UNIFICADOS, 'laudos_rotulados.parquet'))
    print(f"1. Laudos (Rotulados) salvos: {len(df_laudos)} pacientes únicos.")
else:
    print("1. Nenhum Laudo encontrado.")

# --- Medicamentos (Peso/Altura) ---
if lista_medicamentos:
    df_medicamentos = pd.concat(lista_medicamentos, ignore_index=True)

    # convertendo
    df_medicamentos['AM_PESO'] = pd.to_numeric(df_medicamentos['AM_PESO'], errors='coerce')
    df_medicamentos['AM_ALTURA'] = pd.to_numeric(df_medicamentos['AM_ALTURA'], errors='coerce')

    # renomear
    df_medicamentos = df_medicamentos.rename(columns={'AP_CNSPCN': 'CNS', 'AM_PESO': 'PESO', 'AM_ALTURA': 'ALTURA'})

    # remover NaNs
    df_medicamentos = df_medicamentos.dropna(subset=['PESO', 'ALTURA'])

    # 🔥 filtro ANTES da média
    df_medicamentos = df_medicamentos[
        (df_medicamentos['PESO'] > 0) &
        (df_medicamentos['PESO'] <= 400) &
        (df_medicamentos['ALTURA'] > 0) &
        (df_medicamentos['ALTURA'] <= 250)
    ]

    # Agora sim: média por paciente
    df_medicamentos = df_medicamentos.groupby('CNS')[['PESO', 'ALTURA']].mean().reset_index()

    # salvar
    df_medicamentos.to_parquet(os.path.join(PASTA_DADOS_UNIFICADOS, 'medicamentos_features.parquet'))

else:
    print("2. Nenhum dado de Medicamentos (Peso/Altura) encontrado.")

# --- Histórico (Z833) ---
if lista_historico_filtrados:
    df_historico = pd.concat(lista_historico_filtrados, ignore_index=True)
    df_historico = df_historico.rename(columns={'CNS_PAC': 'CNS'})
    df_historico['TEM_HISTORICO_DM'] = 1
    df_historico_final = df_historico[['CNS', 'TEM_HISTORICO_DM']].drop_duplicates(subset=['CNS'])
    df_historico_final.to_parquet(os.path.join(PASTA_DADOS_UNIFICADOS, 'historico_familiar.parquet'))
    print(f"3. Histórico Familiar salvo: {len(df_historico_final)} pacientes.")
else:
    print("3. Nenhum Histórico Familiar (Z833) encontrado.")


# --------------------------------------------------------------------------------
# --- 7. JUNÇÃO FINAL DOS DATASETS ---
# --------------------------------------------------------------------------------

print("\n--- INICIANDO JUNÇÃO FINAL DOS 3 ARQUIVOS (.parquet) ---")

try:
    # Carregar os datasets intermediários
    df_principal = pd.read_parquet(os.path.join(PASTA_DADOS_UNIFICADOS, 'laudos_rotulados.parquet'))
    df_features = pd.read_parquet(os.path.join(PASTA_DADOS_UNIFICADOS, 'medicamentos_features.parquet'))
    df_historico = pd.read_parquet(os.path.join(PASTA_DADOS_UNIFICADOS, 'historico_familiar.parquet'))

    # 7.1. Merge 1: Juntando Principal e Features Clínicas (Peso/Altura)
    df_final = pd.merge(
        df_principal, 
        df_features, 
        on='CNS', 
        how='inner'
    )

    # 7.2. Merge 2: Juntando Histórico Familiar
    df_final = pd.merge(
        df_final, 
        df_historico, 
        on='CNS', 
        how='left'
    )
    
   # --- Criar IMC (PESO / ALTURA²) ---
    df_final['IMC'] = df_final.apply(
    lambda x: x['PESO'] / ((x['ALTURA'] / 100) ** 2)
    if pd.notna(x['PESO']) and pd.notna(x['ALTURA']) and x['PESO'] > 0 and x['ALTURA'] > 0
    else None,
    axis=1
)


    # 7.3. Tratamento Inicial de Valores Faltantes
    
    # Histórico Familiar: NaN significa SEM HISTÓRICO (0)
    df_final['TEM_HISTORICO_DM'] = df_final['TEM_HISTORICO_DM'].fillna(0).astype(int)
    

    # Raça/Cor: Imputa '99' (Sem Informação) para faltantes, mantendo o tipo string/object para o One-Hot Encoding futuro.
    df_final['RACA_COR'] = df_final['RACA_COR'].fillna('99').astype(str)
    
    # Sexo: Imputa '9' (Não Informado) para faltantes, para consistência.
    df_final['SEXO'] = df_final['SEXO'].fillna('9').astype(str) 


    # 7.4. Tratamento Inicial de Valores Faltantes
   
    df_final = df_final[
    (df_final['ALTURA'] > 100) &
    (df_final['PESO'] > 20)
    ]

    #  Recalcular IMC com altura em metros

    df_final['IMC'] = df_final.apply(
    lambda x: x['PESO'] / ((x['ALTURA'] / 100) ** 2)
    if x['ALTURA'] > 0 else np.nan,
    axis=1
    )

    # Reclassifica IMC

    def classifica_imc(imc):
        if pd.isna(imc): return None
        if imc < 18.5: return "Abaixo do peso"
        if imc < 25: return "Normal"
        if imc < 30: return "Sobrepeso"
        return "Obesidade"

    df_final['IMC_CLASS'] = df_final['IMC'].apply(classifica_imc)

#  Remover IMCs fora da realidade

    df_final = df_final[(df_final['IMC'] >= 10) & (df_final['IMC'] <= 60)]

        #  Remover pacientes sem informações essenciais


    df_final = df_final.dropna(subset=['IDADE', 'SEXO', 'CID_DIAGNOSTICO'])


    # 7.5. Salvar o Dataset Final
    df_final.to_parquet(ARQUIVO_FINAL_MODELO)
    df_final["ID"] = range(1, len(df_final)+1)
    df_final = df_final.drop(columns=["CNS"])
    df_final.to_parquet(...)


    print("\n--- DATASET FINAL DE MODELAGEM CRIADO ---")
    print(f"Arquivo final salvo em: {ARQUIVO_FINAL_MODELO}")
    print(f"Total de Pacientes no Dataset Final: {len(df_final)}")
    
    print("\nStatus de Valores Faltantes (NaN) nas features principais após Junção:")
    # Verifica a porcentagem de NaNs para Peso, Altura e Raca/Cor
   print("\n--- PORCENTAGEM DE DADOS FALTANTES EM TODAS AS FEATURES ---")
print((df_final.isnull().sum() / len(df_final) * 100).sort_values(ascending=False))


except Exception as e:
    print(f"\nERRO na fase de Junção Final: {e}")
    print("Verifique se os arquivos intermediários foram criados corretamente na Seção 6.")


print("\n--- SCRIPT 1 (UNIFICAÇÃO COMPLETA) CONCLUÍDO ---")