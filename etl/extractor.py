import pandas as pd
import glob
import os
import pyarrow.parquet as pq
from typing import List, Tuple
import config  # Importa as configurações

class DataExtractor:
    def __init__(self):
        self.raw_dir = config.RAW_DIR

    def get_medicamentos_data(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Lê arquivos AMSP e retorna duas listas: 
        uma com dados de rótulos/demográficos e outra com features (peso/altura).
        """
        pattern = os.path.join(self.raw_dir, 'AMSP*.parquet')
        files = glob.glob(pattern)
        print(f"--- Processando {len(files)} arquivos de Medicamentos (AMSP) ---")

        lista_rotulos = []
        lista_features = []

        for f in files:
            try:
                df = pd.read_parquet(f, columns=config.COLS_AMSP)
                
                # Separação lógica das responsabilidades
                # 1. Parte Demográfica + Rótulo
                df_rot = df[['AP_CNSPCN', 'AP_PRIPAL', 'AP_NUIDADE', 'AP_SEXO', 'AP_RACACOR']].copy()
                lista_rotulos.append(df_rot)

                # 2. Parte Física (Features)
                df_feat = df[['AP_CNSPCN', 'AM_PESO', 'AM_ALTURA']].copy()
                lista_features.append(df_feat)
                
            except Exception as e:
                print(f"Erro ao ler {f}: {e}")

        return lista_rotulos, lista_features

    def get_historico_data(self) -> List[pd.DataFrame]:
        """
        Lê arquivos BISP usando fragmentos para economizar memória.
        """
        pattern = os.path.join(self.raw_dir, 'BISP*.parquet')
        files = glob.glob(pattern)
        print(f"--- Processando {len(files)} diretórios de Histórico (BISP) ---")

        lista_hist = []

        for directory in files:
            try:
                dataset = pq.ParquetDataset(directory)
                # Itera sobre fragmentos para evitar OOM (Out of Memory)
                for fragment in dataset.fragments:
                    df_chunk = fragment.to_table(columns=config.COLS_BISP).to_pandas()
                    
                    # Filtra imediatamente para reduzir tamanho
                    filtro = df_chunk['CIDPRI'].str.strip().str.upper() == config.CID_HISTORICO
                    df_filtered = df_chunk[filtro]
                    
                    if not df_filtered.empty:
                        lista_hist.append(df_filtered[['CNS_PAC']])
            except Exception as e:
                print(f"Erro ao ler diretório {directory}: {e}")
        
        return lista_hist
