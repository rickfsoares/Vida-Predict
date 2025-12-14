# run_experiments.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import config
import os


def run_regression_benchmark():
    print("\n========================================================")
    print("🧪 INICIANDO EXPERIMENTOS DE REGRESSÃO (BENCHMARK)")
    print("========================================================")
    print("Objetivo: Tentar prever o PESO usando apenas ALTURA, IDADE e SEXO.")
    print("Hipótese: Se o R² for alto, o PESO é redundante e valida a escolha do artigo.")

    # 1. Carregar Dataset Completo (Features + Peso + Target)
    if not os.path.exists(config.FILE_FEATURES_COMPLETO):
        print(f"ERRO: Arquivo {config.FILE_FEATURES_COMPLETO} não encontrado.")
        print("Rode 'python pipeline.py' primeiro.")
        return

    df = pd.read_parquet(config.FILE_FEATURES_COMPLETO)

    # 2. Preparar Dados
    # Remove linhas onde Peso/Altura/Idade possam ser nulos (segurança extra para a regressão)
    df_reg = df.dropna(subset=['PESO', 'ALTURA', 'IDADE'])

    # Define Features (X) e Target (y)
    # Seleciona ALTURA, IDADE e todas as colunas de SEXO (que foram geradas pelo One-Hot Encoding)
    # Nota: Não usamos 'TEM_DM1' aqui, pois queremos provar relações físicas, não a doença.
    features = ['ALTURA', 'IDADE'] + \
        [c for c in df_reg.columns if 'SEXO_' in c]

    X = df_reg[features]
    y = df_reg['PESO']

    print(f"\nDataset de Treino: {X.shape}")
    print(f"Features usadas: {features}")

    # 3. Divisão Treino/Teste
    # Usamos 30% para teste e fixamos random_state para reprodutibilidade
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 4. Definir Modelos para Comparação
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    }

    # 5. Executar Benchmark
    print("\n--- RESULTADOS ---")
    print(f"{'MODELO':<20} | {'R²':<10} | {
          'RMSE (kg)':<10} | {'MAE (kg)':<10}")
    print("-" * 60)

    melhor_r2 = -1
    melhor_modelo = ""

    for nome, modelo in modelos.items():
        # Treinar
        modelo.fit(X_train, y_train)

        # Predizer
        y_pred = modelo.predict(X_test)

        # Calcular Métricas
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"{nome:<20} | {r2:.4f}     | {rmse:.2f}       | {mae:.2f}")

        if r2 > melhor_r2:
            melhor_r2 = r2
            melhor_modelo = nome

    print("-" * 60)
    print("\n>>> CONCLUSÃO AUTOMÁTICA:")

    # Regra de decisão para o relatório
    if melhor_r2 > 0.5:
        print(f"✅ SUCESSO: O modelo {melhor_modelo} conseguiu explicar {
              melhor_r2*100:.1f}% da variação do peso.")
        print("Isso confirma estatisticamente que o PESO é fortemente dependente da Altura/Idade/Sexo.")
        print("A remoção da coluna PESO para o modelo de classificação final (Artigo) é JUSTIFICADA.")
    else:
        print("⚠️ ALERTA: A correlação encontrada foi baixa. Reavaliar a remoção do Peso.")


if __name__ == "__main__":
    run_regression_benchmark()
