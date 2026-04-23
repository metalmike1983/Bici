# =========================================
# 1. IMPORT
# =========================================
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# =========================================
# 2. CONFIG
# =========================================
TARGET = "Cluster"   # tappeto = 1
ID_COL = "ndg"

# =========================================
# 3. FEATURE ENGINEERING
# =========================================

df = df.copy()

# --- CAPACITA' DI RIENTRO
df["CAPACITA_RIENTRO"] = df["CC_IMP_STIPEND_AVG_3M"] - df["F_IMP_SCONFNTO_TOT_UM"]

# --- STRESS
df["STRESS_RATIO"] = df["F_IMP_SCONFNTO_TOT_UM"] / (df["CC_IMP_STIPEND_AVG_3M"] + 1)

# --- LIQUIDITA'
df["LIQUIDITA_NETTA"] = df["CC_SALDO_AVG_3M"] - df["F_IMP_SCONFNTO_TOT_UM"]

# --- ATTIVITA'
df["ATTIVITA_3M"] = (
    df["CC_IMP_CONTANTI_AVG_3M"] +
    df["CC_IMP_UTENZE_AVG_3M"]
)

# --- LEVA (già esiste ma rinominiamo per chiarezza)
df["LEVA_CF"] = df["F_PER_UTILZ_CF_UM"]

# =========================================
# 4. FEATURE SELECTION
# =========================================

drop_cols = [
    TARGET,
    ID_COL,
    "MESE"   # opzionale
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[TARGET]

# =========================================
# 5. IMPUTAZIONE
# =========================================

X = X.fillna(X.median(numeric_only=True))

# =========================================
# 6. TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================================
# 7. MODELLO XGBOOST
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

model.fit(X_train, y_train)

# =========================================
# 8. PREDIZIONE
# =========================================

y_proba = model.predict_proba(X_test)[:,1]

# soglia ottimizzata per recall tappeto
threshold = 0.40
y_pred = (y_proba > threshold).astype(int)

# =========================================
# 9. METRICHE
# =========================================

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("\n=== ROC AUC ===")
print(roc_auc_score(y_test, y_proba))

# =========================================
# 10. FEATURE IMPORTANCE
# =========================================

feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== TOP FEATURE ===")
print(feat_imp.head(20))

# =========================================
# 11. SCORING COMPLETO
# =========================================

df_scored = df.copy()
df_scored["prob_tappeto"] = model.predict_proba(X)[:,1]

# =========================================
# 12. SEGMENTAZIONE BUSINESS
# =========================================

def segment(x):
    if x >= 0.7:
        return "AUTO"        # si autoregola
    elif x >= 0.4:
        return "SOFT"        # reminder
    else:
        return "HARD"        # azione forte

df_scored["segmento"] = df_scored["prob_tappeto"].apply(segment)

# =========================================
# 13. OUTPUT TOP CLIENTI
# =========================================

top = df_scored[[ID_COL, "prob_tappeto", "segmento"]]\
        .sort_values("prob_tappeto", ascending=False)

print("\n=== TOP CLIENTI ===")
print(top.head(20))
