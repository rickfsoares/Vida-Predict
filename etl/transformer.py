import pandas as pd
import numpy as np
import config


class DataTransformer:

    @staticmethod
    def process_rotulos(lista_dfs: list) -> pd.DataFrame:
        if not lista_dfs:
            return pd.DataFrame()

        df = pd.concat(lista_dfs, ignore_index=True)

        # Renomeação e Criação do Target
        df = df.rename(columns={
            'AP_CNSPCN': 'CNS', 'AP_PRIPAL': 'CID_DIAGNOSTICO',
            'AP_NUIDADE': 'IDADE', 'AP_SEXO': 'SEXO', 'AP_RACACOR': 'RACA_COR'
        })

        # Converte IDADE para número
        df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')

        # Criação do Target Binário
        df['TEM_DM1'] = df['CID_DIAGNOSTICO'].str.startswith(
            config.CID_ALVO_DM1, na=False).astype(int)

        # Remove duplicatas de pacientes
        return df.drop_duplicates(subset=['CNS'], keep='first')

    @staticmethod
    def process_features_fisicas(lista_dfs: list) -> pd.DataFrame:
        if not lista_dfs:
            return pd.DataFrame()

        df = pd.concat(lista_dfs, ignore_index=True)

        # Limpeza Numérica
        df['AM_PESO'] = pd.to_numeric(df['AM_PESO'], errors='coerce')
        df['AM_ALTURA'] = pd.to_numeric(df['AM_ALTURA'], errors='coerce')

        df = df.rename(columns={'AP_CNSPCN': 'CNS',
                       'AM_PESO': 'PESO', 'AM_ALTURA': 'ALTURA'})
        df = df.dropna(subset=['PESO', 'ALTURA'])

        # Agregação (Média por paciente)
        return df.groupby('CNS')[['PESO', 'ALTURA']].mean().reset_index()

    @staticmethod
    def process_historico(lista_dfs: list) -> pd.DataFrame:
        if not lista_dfs:
            return pd.DataFrame(columns=['CNS', 'TEM_HISTORICO_DM'])

        df = pd.concat(lista_dfs, ignore_index=True)
        df = df.rename(columns={'CNS_PAC': 'CNS'})
        df['TEM_HISTORICO_DM'] = 1
        return df[['CNS', 'TEM_HISTORICO_DM']].drop_duplicates()

    @staticmethod
    def engineer_final_dataset(df_rotulos, df_features, df_historico) -> pd.DataFrame:
        """Realiza o merge e cálculos finais (IMC)"""

        print("--- Realizando Merge e Engenharia de Features ---")

        # 1. Merge Left (Mantém base de pacientes do Rótulo)
        df_final = pd.merge(df_rotulos, df_features, on='CNS', how='left')
        df_final = pd.merge(df_final, df_historico, on='CNS', how='left')

        # 2. Imputações Básicas
        df_final['TEM_HISTORICO_DM'] = df_final['TEM_HISTORICO_DM'].fillna(
            0).astype(int)
        df_final['RACA_COR'] = df_final['RACA_COR'].fillna('99').astype(str)
        df_final['SEXO'] = df_final['SEXO'].fillna('9').astype(str)

        # 3. Cálculo do IMC
        # Evita divisão por zero e converte cm -> m
        df_final['IMC'] = df_final.apply(
            lambda row: row['PESO'] / ((row['ALTURA'] / 100) ** 2)
            if pd.notnull(row['ALTURA']) and row['ALTURA'] > 0 else np.nan, axis=1
        )

        # 4. Classificação Categórica do IMC (Opcional, mas útil para análise)
        def classifica_imc(imc):
            if pd.isna(imc):
                return None
            if imc < 18.5:
                return "Abaixo do peso"
            if imc < 25:
                return "Normal"
            if imc < 30:
                return "Sobrepeso"
            return "Obesidade"

        df_final['IMC_CLASS'] = df_final['IMC'].apply(classifica_imc)

        # 5. Filtragem de Outliers Biológicos
        df_final = df_final[
            (df_final['IMC'] >= config.IMC_MIN) &
            (df_final['IMC'] <= config.IMC_MAX)
        ]

        # 6. Criação de ID Numérico (Anonimização final para o modelo)
        df_final["ID_PACIENTE"] = range(1, len(df_final) + 1)

        # Remove a coluna CNS para evitar vazamento de dados no futuro,
        # mas mantém se precisar debugar. O ideal é remover antes do treino.
        # df_final = df_final.drop(columns=["CNS"])

        return df_final
