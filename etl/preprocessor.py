import pandas as pd
import config


class ModelPreprocessor:
    """
    Prepara os dados para o modelo (One-Hot Encoding, Casting, etc)
    """

    def apply_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Aplicando Pré-processamento (One-Hot Encoding) ---")

        # Garantir tipos categóricos (como feito pelo colega)
        df['RACA_COR'] = df['RACA_COR'].astype(str)

        # Se IMC_CLASS ainda não existe ou precisa ser reforçado
        if 'IMC_CLASS' in df.columns:
            df['IMC_CLASS'] = df['IMC_CLASS'].astype(str)
            cols_to_encode = ['RACA_COR', 'IMC_CLASS']
        else:
            cols_to_encode = ['RACA_COR']

        # Aplica One-Hot Encoding
        df_ohe = pd.get_dummies(
            df,
            columns=cols_to_encode,
            prefix=cols_to_encode,
            drop_first=False
        )

        print(f"Colunas geradas: {list(df_ohe.columns)}")
        return df_ohe
