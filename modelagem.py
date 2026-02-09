import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

from imblearn.combine import SMOTETomek

# ----------------------------------------------------------------------
# 1. CONFIGURAÇÕES BÁSICAS
# ----------------------------------------------------------------------

DATASET_PATH = "./dados-unificados/dataset_modelagem_ohe.parquet"
TARGET = "TEM_DM1"
RANDOM_STATE = 42

print("\n====================================================")
print("🚀 INICIANDO ETAPA DE MODELAGEM PREDITIVA")
print("====================================================")

# ----------------------------------------------------------------------
# 2. CARGA DO DATASET
# ----------------------------------------------------------------------

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Arquivo não encontrado: {DATASET_PATH}")

df = pd.read_parquet(DATASET_PATH)
print(f"Dataset carregado: {df.shape}")

# ----------------------------------------------------------------------
# 3. DEFINIÇÃO DE FEATURES E TARGET
# ----------------------------------------------------------------------

if TARGET not in df.columns:
    raise ValueError(f"Coluna target '{TARGET}' não encontrada no dataset")

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nDistribuição absoluta do target (antes da limpeza):")
print(y.value_counts())

# ----------------------------------------------------------------------
# 4. LIMPEZA DE NaN E VALORES INFINITOS  ✅ (ALTERAÇÃO APLICADA)
# ----------------------------------------------------------------------

print("\n🧹 Limpando NaNs e valores infinitos antes do SMOTE...")

# Substitui ±inf por NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

antes = X.shape[0]

# Remove linhas com qualquer NaN
mask = ~X.isna().any(axis=1)
X = X.loc[mask]
y = y.loc[mask]

depois = X.shape[0]

print(f"Registros removidos por NaN/inf: {antes - depois}")
print(f"Dataset limpo: {X.shape}")

print("\nDistribuição do target (após limpeza):")
print(y.value_counts())

# 🔴 PROTEÇÃO CRÍTICA
if y.nunique() < 2:
    raise ValueError(
        "❌ O target TEM_DM1 possui apenas uma classe após a limpeza. "
        "Não é possível aplicar SMOTE nem treinar um classificador."
    )

# ----------------------------------------------------------------------
# 5. GARANTIA DE FEATURES NUMÉRICAS
# ----------------------------------------------------------------------

# Manter apenas colunas numéricas
X = X.select_dtypes(include=["number"])

# Remover colunas constantes (zero variância)
colunas_constantes = [c for c in X.columns if X[c].nunique() <= 1]
if colunas_constantes:
    print("⚠️ Removendo colunas constantes:", colunas_constantes)
    X = X.drop(columns=colunas_constantes)

print(f"\nTotal de features usadas: {X.shape[1]}")

# ----------------------------------------------------------------------
# 6. SPLIT TREINO / TESTE
# ----------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nSplit realizado:")
print("Treino:", X_train.shape)
print("Teste :", X_test.shape)

# ----------------------------------------------------------------------
# 7. BALANCEAMENTO COM SMOTE + TOMEK (APENAS TREINO)
# ----------------------------------------------------------------------

print("\n⚖️ Aplicando SMOTE + Tomek Links no conjunto de treino...")

smote_tomek = SMOTETomek(random_state=RANDOM_STATE)

X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)

print("Distribuição após balanceamento:")
print(pd.Series(y_train_bal).value_counts())

# ----------------------------------------------------------------------
# 8. TREINAMENTO DO MODELO
# ----------------------------------------------------------------------

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\n🧠 Treinando modelo Random Forest...")
modelo.fit(X_train_bal, y_train_bal)

# ----------------------------------------------------------------------
# 9. AVALIAÇÃO NO CONJUNTO DE TESTE
# ----------------------------------------------------------------------

print("\n📊 Avaliando modelo no conjunto de teste...")

y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print("\n--- MÉTRICAS ---")
print(f"Acurácia : {acc:.4f}")
print(f"Precisão : {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc:.4f}")

# ----------------------------------------------------------------------
# 10. MATRIZ DE CONFUSÃO
# ----------------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n--- MATRIZ DE CONFUSÃO ---")
print(cm)

print("\n====================================================")
print("✅ MODELAGEM FINALIZADA COM SUCESSO")
print("====================================================")
