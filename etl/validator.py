import pandas as pd
import os
import config


class DataValidator:
    """
    Responsável por gerar relatórios de inconsistências (QA)
    """

    def __init__(self):
        self.output_dir = config.REPORTS_DIR

    def gerar_relatorios_absurdos(self, df: pd.DataFrame):
        print("\n--- Executando Validação de Qualidade de Dados (QA) ---")

        absurdos_gerais = []

        # 1. Verificações Gerais (Peso e Altura Impossíveis)
        checks = [
            (df["PESO"] <= 0, "Peso <= 0"),
            (df["PESO"] > 400, "Peso > 400"),
            (df["ALTURA"] <= 0, "Altura <= 0"),
            (df["ALTURA"] > 250, "Altura > 250")
        ]

        for condition, label in checks:
            inconsistencias = df[condition]
            if not inconsistencias.empty:
                absurdos_gerais.append(
                    inconsistencias.head(30).assign(PROBLEMA=label))

        # Salva Absurdos Gerais
        if absurdos_gerais:
            df_geral = pd.concat(absurdos_gerais, ignore_index=True)
            path = os.path.join(self.output_dir, "absurdos_gerais.csv")
            df_geral.to_csv(path, index=False)
            print(f"[QA] Relatório salvo: {path}")

        # 2. Verificações por Faixa Etária

        # Crianças / RN
        df_rn_abs = df[(df["IDADE"] <= 11) & (
            (df["ALTURA"] > 180) | (df["PESO"] > 120))]
        if not df_rn_abs.empty:
            path = os.path.join(self.output_dir, "absurdos_criancas_rn.csv")
            df_rn_abs.to_csv(path, index=False)
            print(f"[QA] Relatório salvo: {path}")

        # Adolescentes
        df_ado_abs = df[(df["IDADE"] >= 12) & (df["IDADE"] <= 17) & (
            (df["ALTURA"] > 220) | (df["PESO"] > 200))]
        if not df_ado_abs.empty:
            path = os.path.join(self.output_dir, "absurdos_adolescentes.csv")
            df_ado_abs.to_csv(path, index=False)
            print(f"[QA] Relatório salvo: {path}")

        # Adultos
        df_adult_abs = df[(df["IDADE"] >= 18) & (
            (df["ALTURA"] > 250) | (df["PESO"] > 400))]
        if not df_adult_abs.empty:
            path = os.path.join(self.output_dir, "absurdos_adultos.csv")
            df_adult_abs.to_csv(path, index=False)
            print(f"[QA] Relatório salvo: {path}")
