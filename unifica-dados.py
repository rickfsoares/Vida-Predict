import pandas as pd
import os
import glob
import pyarrow.parquet as pq

# --- 1. DEFINIÇÕES ---
PASTA_DADOS_BRUTOS_PARQUET = 'dados-datasus-processados'
PASTA_DADOS_UNIFICADOS = 'dados-unificados'
os.makedirs(PASTA_DADOS_UNIFICADOS, exist_ok=True)

CID_ALVO_DM1 = 'E10' 
CID_HISTORICO_FAMILIAR = 'Z833'

# --- 2. LISTAS PARA COLETAR OS "CHUNKS" ---
lista_laudos_rotulados = []
lista_medicamentos = []
lista_historico_filtrados = []

print("--- INICIANDO UNIFICAÇÃO (COM RÓTULOS 0 E 1) ---")

# --- 3. COLETANDO TUDO (RÓTULOS E FEATURES) A PARTIR DE AMSP ---
colunas_necessarias_amsp = ['AP_CNSPCN', 'AP_PRIPAL', 'AP_NUIDADE', 'AP_SEXO', 'AM_PESO', 'AM_ALTURA']

padrao_am = os.path.join(PASTA_DADOS_BRUTOS_PARQUET, 'AMSP*.parquet') 
arquivos_am = glob.glob(padrao_am)
print(f"\nProcessando {len(arquivos_am)} arquivos de Medicamentos (AM) para Rótulos e Features...")

for parquet_file in arquivos_am:
    try:
        # Lê todas as colunas necessárias de uma vez
        df_chunk = pd.read_parquet(parquet_file, columns=colunas_necessarias_amsp)
        
        # 1. RÓTULOS (Para lista_laudos_rotulados)
        df_rotulos = df_chunk[['AP_CNSPCN', 'AP_PRIPAL', 'AP_NUIDADE', 'AP_SEXO']].copy()
        df_rotulos['TEM_DM1'] = df_rotulos['AP_PRIPAL'].str.startswith(CID_ALVO_DM1, na=False).astype(int)
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


# --- 5. HISTÓRICO (BI) ---
colunas_historico_necessarias = ['CNS_PAC', 'CIDPRI']
padrao_bi = os.path.join(PASTA_DADOS_BRUTOS_PARQUET, 'BISP*.parquet')
arquivos_bi = glob.glob(padrao_bi)

print(f"\nProcessando {len(arquivos_bi)} diretórios de BPA-I (BI)...")

for parquet_directory in arquivos_bi:
    try:
        # MODO SEGURO: Processa fragmentos/arquivos individualmente
        print(f"  Processando {os.path.basename(parquet_directory)} em fragmentos (modo seguro)...")
        
        dataset = pq.ParquetDataset(parquet_directory)
        
        # Iterar sobre os fragments (arquivos individuais) do Dataset
        for fragment in dataset.fragments:
            
            # Lendo cada fragmento de forma independente e convertendo para pandas
            df_chunk = fragment.to_table(columns=colunas_historico_necessarias).to_pandas()
            
            # Filtro
            filtro_hist = df_chunk['CIDPRI'].str.strip().str.upper() == CID_HISTORICO_FAMILIAR
            df_filtrado = df_chunk[filtro_hist]
            
            if not df_filtrado.empty:
                lista_historico_filtrados.append(df_filtrado[['CNS_PAC']]) 
                
        print(f"  -> {os.path.basename(parquet_directory)} concluído.")

    except Exception as e:
        print(f"  -> Erro ao processar {parquet_directory}: {e}")

# --- 6. UNIFICAR E SALVAR ---
print("\n--- Unificando e Salvando Arquivos ---")
print(f"Chunks de Laudos (Rótulos): {len(lista_laudos_rotulados)}")
print(f"Chunks de Features (Peso/Altura): {len(lista_medicamentos)}")
print(f"Chunks de Histórico (Z833): {len(lista_historico_filtrados)}")

# --- Laudos (Rotulados) ---
if lista_laudos_rotulados:
    df_laudos = pd.concat(lista_laudos_rotulados, ignore_index=True)
    
    df_laudos = df_laudos.rename(columns={'AP_CNSPCN': 'CNS', 'AP_PRIPAL': 'CID_DIAGNOSTICO',
                                          'AP_NUIDADE': 'IDADE', 'AP_SEXO': 'SEXO'})
                                          
    df_laudos = df_laudos.drop_duplicates(subset=['CNS'], keep='first')
    df_laudos.to_parquet(os.path.join(PASTA_DADOS_UNIFICADOS, 'laudos_rotulados.parquet'))
    print(f"Laudos (Rotulados) salvos: {len(df_laudos)} pacientes únicos.")
    print("Distribuição das classes no arquivo de Laudos:")
    print(df_laudos['TEM_DM1'].value_counts(normalize=True))
else:
    print("Nenhum Laudo encontrado.")

# --- Medicamentos (Peso/Altura) ---
if lista_medicamentos:
    df_medicamentos = pd.concat(lista_medicamentos, ignore_index=True)

    # Forçar 'AM_PESO' e 'AM_ALTURA' a serem numéricos.
    df_medicamentos['AM_PESO'] = pd.to_numeric(df_medicamentos['AM_PESO'], errors='coerce')
    df_medicamentos['AM_ALTURA'] = pd.to_numeric(df_medicamentos['AM_ALTURA'], errors='coerce')

    # Renomear AP_CNSPCN, AM_PESO, AM_ALTURA
    df_medicamentos = df_medicamentos.rename(columns={'AP_CNSPCN': 'CNS', 'AM_PESO': 'PESO', 'AM_ALTURA': 'ALTURA'})
    
    df_medicamentos = df_medicamentos.dropna(subset=['PESO', 'ALTURA'])

    df_medicamentos = df_medicamentos.groupby('CNS')[['PESO', 'ALTURA']].mean().reset_index()
    df_medicamentos.to_parquet(os.path.join(PASTA_DADOS_UNIFICADOS, 'medicamentos_features.parquet'))
    print(f"Features (Peso/Altura) salvas: {len(df_medicamentos)} registros.")
else:
    print("Nenhum dado de Medicamentos (Peso/Altura) encontrado.")

# --- Histórico (Z833) ---
if lista_historico_filtrados:
    df_historico = pd.concat(lista_historico_filtrados, ignore_index=True)
    df_historico = df_historico.rename(columns={'CNS_PAC': 'CNS'})
    df_historico['TEM_HISTORICO_DM'] = 1
    df_historico_final = df_historico[['CNS', 'TEM_HISTORICO_DM']].drop_duplicates(subset=['CNS'])
    df_historico_final.to_parquet(os.path.join(PASTA_DADOS_UNIFICADOS, 'historico_familiar.parquet'))
    print(f"Histórico Familiar salvo: {len(df_historico_final)} pacientes.")
else:
    print("Nenhum Histórico Familiar (Z833) encontrado.")

print("\n--- SCRIPT 1 (UNIFICAÇÃO) CONCLUÍDO ---")
