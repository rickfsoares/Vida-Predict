import os

# Caminhos
RAW_DIR = 'dados-datasus-processados'
PROCESSED_DIR = 'dados-unificados'
REPORTS_DIR = 'relatorios-consistencia'
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Arquivos de Saída (Parquet)
FILE_LAUDOS = os.path.join(PROCESSED_DIR, 'laudos_rotulados.parquet')
FILE_FEATURES = os.path.join(PROCESSED_DIR, 'medicamentos_features.parquet')
FILE_HISTORICO = os.path.join(PROCESSED_DIR, 'historico_familiar.parquet')
FILE_FINAL = os.path.join(PROCESSED_DIR, 'dataset_final_modelo.parquet')
FILE_FEATURES_COMPLETO = os.path.join(
    PROCESSED_DIR, 'dataset_features_completo.parquet')

# Constantes de Negócio
CID_ALVO_DM1 = 'E10'
CID_HISTORICO = 'Z833'

# Colunas Necessárias
COLS_AMSP = ['AP_CNSPCN', 'AP_PRIPAL', 'AP_NUIDADE',
             'AP_SEXO', 'AP_RACACOR', 'AM_PESO', 'AM_ALTURA']
COLS_BISP = ['CNS_PAC', 'CIDPRI']

# Limites
IMC_MIN = 10
IMC_MAX = 60
