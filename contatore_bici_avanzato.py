# =========================================
# 1. IMPORT
# =========================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier

# =========================================
# 2. CONFIG
# =========================================
TARGET = "Cluster"
ID_COL = "ndg"

# =========================================
# 3. FEATURE ENGINEERING
# =========================================

df = df.copy()

df["CAPACITA_RIENTRO"] = df["CC_IMP_STIPEND_AVG_3M"] - df["F_IMP_SCONFNTO_TOT_UM"]
df["STRESS_RATIO"] = df["F_IMP_SCONFNTO_TOT_UM"] / (df["CC_IMP_STIPEND_AVG_3M"] + 1)
df["LIQUIDITA_NETTA"] = df["CC_SALDO_AVG_3M"] - df["F_IMP_SCONFNTO_TOT_UM"]
df["ATTIVITA_3M"] = df["CC_IMP_CONTANTI_AVG_3M"] + df["CC_IMP_UTENZE_AVG_3M"]
df["LEVA_CF"] = df["F_PER_UTILZ_CF_UM"]

# =========================================
# 4. SPLIT X / y
# =========================================

drop_cols = [TARGET, ID_COL, "MESE"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[TARGET]

# =========================================
# 5. COLONNE
# =========================================

num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# =========================================
# 6. PREPROCESSING
# =========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# =========================================
# 7. MODELLO
# =========================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42
)

# =========================================
# 8. PIPELINE
# =========================================

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# =========================================
# 9. TRAIN TEST
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================================
# 10. TRAIN
# =========================================

clf.fit(X_train, y_train)

# =========================================
# 11. EVALUATION
# =========================================

y_proba = clf.predict_proba(X_test)[:,1]

threshold = 0.40
y_pred = (y_proba > threshold).astype(int)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("\n=== ROC AUC ===")
print(roc_auc_score(y_test, y_proba))

# =========================================
# 12. FEATURE IMPORTANCE (opzionale)
# =========================================

model_step = clf.named_steps["model"]
print("\nFeature importance disponibili via model_step.feature_importances_")

# =========================================
# 13. SALVATAGGIO
# =========================================

joblib.dump(clf, "modello_tappeto_xgb.joblib")

print("\nModello salvato")
