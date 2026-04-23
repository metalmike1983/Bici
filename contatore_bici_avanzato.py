# =========================================
# MODELLO SELF-CURE (TAPPETO) - XGBOOST
# =========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

# ==============================
# CONFIG
# ==============================
INPUT_FILE = r"C:\Users\...\cl4.xlsx"
TARGET = "Cluster"
ID_COL = "ndg"

TEST_SIZE = 0.3
RANDOM_STATE = 42

# ==============================
# LOAD
# ==============================
df = pd.read_excel(INPUT_FILE)
df.columns = [c.strip() for c in df.columns]

# ==============================
# TARGET CLEAN
# ==============================
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].notna()]
df[TARGET] = df[TARGET].astype(int)

# ==============================
# CLEAN 9999 / SENTINEL
# ==============================
df = df.replace([9999, -9999], np.nan)

# ==============================
# DROP LEAKAGE (CRITICO)
# ==============================
LEAKAGE = [
    "RATING_MINORE",
    "FASCIA_RISCHIO",
    "COD_FASCIA_RISCHIO",
    "RECIDIVITA",
    "FLAG_RECIDIVO",
    "S_TREND_STATUS"
]

df = df.drop(columns=[c for c in LEAKAGE if c in df.columns], errors="ignore")

# ==============================
# FEATURE ENGINEERING (CHIAVE)
# ==============================

# protezione divisioni
def safe_div(a, b):
    return a / (b + 1e-6)

# esempio colonne (adatta ai tuoi nomi reali)
if "SALDO" in df.columns and "IMP_RATA_MENS" in df.columns:
    df["buffer"] = safe_div(df["SALDO"], df["IMP_RATA_MENS"])

if "IMP_UTILZ_TOT" in df.columns and "SALDO" in df.columns:
    df["util_ratio"] = safe_div(df["IMP_UTILZ_TOT"], df["SALDO"])

if "NUM_STIPENDI_3M" in df.columns:
    df["salary_regularity"] = df["NUM_STIPENDI_3M"] / 3

if "SALDO" in df.columns and "SALDO_3M" in df.columns:
    df["saldo_trend"] = df["SALDO"] - df["SALDO_3M"]

# ==============================
# FEATURE SET
# ==============================
exclude_cols = [TARGET, ID_COL]
features = [c for c in df.columns if c not in exclude_cols]

X = df[features].copy()
y = df[TARGET].copy()

# ==============================
# NUMERIC ONLY (XGBoost friendly)
# ==============================
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# imputazione semplice
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ==============================
# MODELLO XGBOOST
# ==============================
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("\n=== ROC AUC ===")
print(roc_auc_score(y_test, y_prob))

# ==============================
# FEATURE IMPORTANCE
# ==============================
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== TOP FEATURE ===")
print(importance.head(20))

# ==============================
# SCORING COMPLETO
# ==============================
df["prob_tappeto"] = model.predict_proba(X)[:, 1]
df = df.sort_values("prob_tappeto", ascending=False)

print("\n=== TOP CLIENTI ===")
print(df[[ID_COL, "prob_tappeto"]].head(20))
