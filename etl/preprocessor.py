import pandas as pd
import config

class ModelPreprocessor:
    """
    Prepara os dados para o modelo (One-Hot Encoding, Casting, etc)
    """

    def apply_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- ⚙️ Aplicando Pré-processamento (One-Hot Encoding) ---")

        # 1. Garantir tipos categóricos (Cast para string)
        df['RACA_COR'] = df['RACA_COR'].astype(str)
        
        # --- CORREÇÃO: Garantir que SEXO é string ---
        df['SEXO'] = df['SEXO'].astype(str) 
        
       # Remover registros com sexo inválido (manter apenas M e F)
        antes = len(df)
        df = df[df['SEXO'].isin(['M', 'F'])].copy()
        depois = len(df)
        print(f"Removidos {antes - depois} registros com SEXO inválido")


                #  1.1 REMOVER RAÇA NÃO INFORMADA (99)
        antes = len(df)
        df = df[df['RACA_COR'] != '99'].copy()
        depois = len(df)
        print(f"Removidos {antes - depois} registros com RACA_COR = 99")

        # 2. Definir colunas para encoding
        cols_to_encode = ['RACA_COR', 'SEXO']

        # Se IMC_CLASS existe, adiciona também
        if 'IMC_CLASS' in df.columns:
            df['IMC_CLASS'] = df['IMC_CLASS'].astype(str)
            cols_to_encode.append('IMC_CLASS')

        # 3. Aplica One-Hot Encoding
        df_ohe = pd.get_dummies(
            df,
            columns=cols_to_encode,
            prefix=cols_to_encode,
            drop_first=False
        )

        print(f"Colunas geradas (amostra): {list(df_ohe.columns)}")
        return df_ohe
