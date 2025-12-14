from etl.extractor import DataExtractor
from etl.transformer import DataTransformer
from etl.validator import DataValidator
from etl.preprocessor import ModelPreprocessor
import config
import pandas as pd


def main():
    # Instancia as classes
    extractor = DataExtractor()
    transformer = DataTransformer()
    validator = DataValidator()
    preprocessor = ModelPreprocessor()

    # ---------------------------------------------------------
    # 1. ETL
    # ---------------------------------------------------------
    print(">>> FASE 1: ETL")
    raw_rotulos, raw_features = extractor.get_medicamentos_data()
    raw_historico = extractor.get_historico_data()

    df_laudos = transformer.process_rotulos(raw_rotulos)
    df_feat = transformer.process_features_fisicas(raw_features)
    df_hist = transformer.process_historico(raw_historico)

    if df_laudos.empty:
        print("Erro crítico: Sem dados de laudos.")
        return

    df_final = transformer.engineer_final_dataset(df_laudos, df_feat, df_hist)

    # Salva o dataset limpo base
    df_final.to_parquet(config.FILE_FINAL)
    print(f"Dataset Base salvo: {config.FILE_FINAL}")

    # ---------------------------------------------------------
    # 2. VALIDAÇÃO / QA
    # ---------------------------------------------------------
    print("\n>>> FASE 2: QA e Relatórios")
    validator.gerar_relatorios_absurdos(df_final)

    # ---------------------------------------------------------
    # 3. PRÉ-PROCESSAMENTO PARA MODELO
    # ---------------------------------------------------------
    print("\n>>> FASE 3: Preparação para Modelagem")
    df_ohe = preprocessor.apply_one_hot_encoding(df_final)

    # Salva o dataset pronto para a IA
    df_ohe.to_parquet(config.FILE_FEATURES_COMPLETO)
    print(f"Dataset OHE salvo: {config.FILE_FEATURES_COMPLETO}")
    print(f"Shape final: {df_ohe.shape}")


if __name__ == "__main__":
    main()
