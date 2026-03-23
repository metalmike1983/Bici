# =========================================================
# MODELLO V2 - CLUSTER TAPPETO OPERATIVO
# Versione corretta con feature engineering su recidivita'
# =========================================================

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# =========================================================
# 1. CONFIG
# =========================================================
INPUT_FILE = r"C:\Users\mike_\OneDrive - BNP Paribas\Bureau\PITONE\input.xlsx"
SHEET_NAME = 0
OUTPUT_DIR = r"C:\Users\mike_\OneDrive - BNP Paribas\Bureau\PITONE\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "target"      # <-- cambia se serve
ID_COL = "ndg"             # <-- cambia se serve
RANDOM_STATE = 42
TEST_SIZE = 0.30

MODEL_NAME = "modello_cluster_tappeto_operativo_v2.joblib"
SCORED_FILE = "scoring_v2.xlsx"
COEF_FILE = "coefficienti_v2.xlsx"

# soglie operative iniziali
THRESH_NON_TOCCARE = 0.15
THRESH_MONITORARE = 0.45

# =========================================================
# 2. LETTURA DATI
# =========================================================
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
print(f"Dataset letto: {df.shape[0]} righe, {df.shape[1]} colonne")

# uniforma nomi colonne
df.columns = [str(c).strip() for c in df.columns]

if TARGET_COL not in df.columns:
    raise ValueError(f"Colonna target '{TARGET_COL}' non trovata nel dataset.")

if ID_COL not in df.columns:
    print(f"ATTENZIONE: colonna ID '{ID_COL}' non trovata. Procedo senza ID dedicato.")
    df[ID_COL] = np.arange(len(df))

# =========================================================
# 3. FEATURE ENGINEERING V2
# =========================================================
# Qui costruisco feature che correggono il bias su RECIDIVITA_RT

def add_feature_if_exists(df, new_col, condition, true_value=1, false_value=0):
    df[new_col] = np.where(condition, true_value, false_value)
    return df

# -----------------------------
# 3.1 Normalizzazione RECIDIVITA
# -----------------------------
if "RECIDIVITA" in df.columns:
    df["RECIDIVITA"] = df["RECIDIVITA"].astype(str).str.strip().str.upper()
else:
    df["RECIDIVITA"] = "MISSING"

# -----------------------------
# 3.2 Proxy di rientro / miglioramento
# -----------------------------
# Usiamo colonne viste nei tuoi screenshot.
# Se alcune non esistono, le inizializziamo a NaN.
proxy_cols = [
    "CC_TREND_SALDO_3_1",
    "CC_TREND_DELTA_AVERE_DARE_3_1",
    "CC_IMP_DELTA_AVERE_DARE_AVG",
    "MEDIA_SALDO",
    "CC_NUM_STIPEND",
    "CC_IMP_STIPEND",
    "MEDIA_NUM_STIPE",
    "F_IMP_RATA_MENS_TOT",
    "F_IMP_UTILZ_TOT",
    "IND_VR_SALDO_CC_UM_SCONF",
    "R_PD_UM"
]

for c in proxy_cols:
    if c not in df.columns:
        df[c] = np.nan

# coercizione numerica prudente
for c in proxy_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# 3.3 Flag stipendio
# -----------------------------
df["HAS_STIPENDIO"] = (
    (df["CC_NUM_STIPEND"].fillna(0) > 0) |
    (df["MEDIA_NUM_STIPE"].fillna(0) > 0) |
    (df["CC_IMP_STIPEND"].fillna(0) > 0)
).astype(int)

# -----------------------------
# 3.4 Trend positivi / negativi
# -----------------------------
df["TREND_SALDO_POS"] = (df["CC_TREND_SALDO_3_1"].fillna(0) > 0).astype(int)
df["TREND_DELTA_POS"] = (df["CC_TREND_DELTA_AVERE_DARE_3_1"].fillna(0) > 0).astype(int)
df["DELTA_AVERE_DARE_POS"] = (df["CC_IMP_DELTA_AVERE_DARE_AVG"].fillna(0) > 0).astype(int)

# -----------------------------
# 3.5 Score di rientro proxy
# -----------------------------
# Pesi semplici e interpretabili
df["SCORE_RIENTRO_PROXY"] = (
    0.40 * df["CC_TREND_SALDO_3_1"].fillna(0) +
    0.35 * df["CC_TREND_DELTA_AVERE_DARE_3_1"].fillna(0) +
    0.25 * np.sign(df["CC_IMP_DELTA_AVERE_DARE_AVG"].fillna(0))
)

# -----------------------------
# 3.6 Intensita' utilizzo / pressione
# -----------------------------
# utile per distinguere recidivo tecnico vs stress finanziario
df["PRESSIONE_FINANZIARIA_PROXY"] = (
    df["F_IMP_UTILZ_TOT"].fillna(0) +
    df["F_IMP_RATA_MENS_TOT"].fillna(0) +
    np.abs(df["IND_VR_SALDO_CC_UM_SCONF"].fillna(0))
)

# -----------------------------
# 3.7 Rischio alto da PD
# -----------------------------
df["PD_ALTA"] = (df["R_PD_UM"].fillna(0) >= 0.05).astype(int)

# -----------------------------
# 3.8 RT “buono” e RT “cattivo”
# -----------------------------
df["IS_RT"] = (df["RECIDIVITA"] == "RT").astype(int)
df["IS_NR"] = (df["RECIDIVITA"] == "NR").astype(int)

df["RT_BUONO"] = (
    (df["RECIDIVITA"] == "RT") &
    (df["SCORE_RIENTRO_PROXY"] > 0) &
    (df["HAS_STIPENDIO"] == 1) &
    (df["PD_ALTA"] == 0)
).astype(int)

df["RT_CATTIVO"] = (
    (df["RECIDIVITA"] == "RT") &
    (
        (df["SCORE_RIENTRO_PROXY"] <= 0) |
        (df["PD_ALTA"] == 1)
    )
).astype(int)

# -----------------------------
# 3.9 Cliente protetto da flussi
# -----------------------------
df["FLUSSI_PROTETTIVI"] = (
    (df["HAS_STIPENDIO"] == 1) &
    (df["TREND_SALDO_POS"] == 1)
).astype(int)

# -----------------------------
# 3.10 Squilibrio operativo
# -----------------------------
df["SQUILIBRIO_OPERATIVO"] = (
    (df["HAS_STIPENDIO"] == 0) &
    (df["TREND_SALDO_POS"] == 0) &
    (df["TREND_DELTA_POS"] == 0)
).astype(int)

# =========================================================
# 4. PULIZIA COLONNE NON USABILI
# =========================================================
drop_cols = [TARGET_COL]

# evita leakage su output/cluster se gia' presenti
possible_leakage = [
    "Cluster",
    "cluster",
    "classe_operativa",
    "RISCHIO",
    "new_priority",
    "max_pred1",
    "prob",
    "prediction",
    "pred",
    "score_model",
    "score"
]

for c in possible_leakage:
    if c in df.columns and c not in drop_cols:
        drop_cols.append(c)

X = df.drop(columns=drop_cols, errors="ignore").copy()
y = df[TARGET_COL].copy()

# Rimuovo ID dalla modellazione ma lo tengo da parte
id_series = X[ID_COL].copy() if ID_COL in X.columns else pd.Series(np.arange(len(X)), name=ID_COL)
X = X.drop(columns=[ID_COL], errors="ignore")

# =========================================================
# 5. IDENTIFICAZIONE FEATURE NUMERICHE / CATEGORICHE
# =========================================================
numeric_cols = X.select_dtypes(include=[np.number, "int64", "float64", "int32", "float32"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

print("\nNumero feature numeriche:", len(numeric_cols))
print("Numero feature categoriche:", len(categorical_cols))

# =========================================================
# 6. TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, id_series,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y if y.nunique() > 1 else None
)

# =========================================================
# 7. PREPROCESSING
# =========================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# compatibilita' sklearn nuova/vecchia
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", ohe)
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)

# =========================================================
# 8. MODELLO
# =========================================================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    ))
])

# =========================================================
# 9. TRAIN
# =========================================================
model.fit(X_train, y_train)

# =========================================================
# 10. VALUTAZIONE TEST
# =========================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 90)
print("VALUTAZIONE TEST SET")
print("=" * 90)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

try:
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC: {auc:.6f}")
except Exception as e:
    print(f"\nROC AUC non calcolabile: {e}")

# =========================================================
# 11. CLASSI OPERATIVE
# =========================================================
# V2: la decisione usa probabilita' + correzione business su RT_BUONO

def assegna_classe_operativa(prob, rt_buono, pd_alta, flussi_protettivi):
    # override protettivo per recidivi tecnici "buoni"
    if rt_buono == 1 and pd_alta == 0 and flussi_protettivi == 1 and prob < 0.50:
        return "NON_TOCCARE_ORA"

    if prob < THRESH_NON_TOCCARE:
        return "NON_TOCCARE_ORA"
    elif prob < THRESH_MONITORARE:
        return "MONITORARE"
    else:
        return "INTERVENIRE_PRIMA"

# test scored
test_scored = X_test.copy()
test_scored[ID_COL] = id_test.values
test_scored[TARGET_COL] = y_test.values
test_scored["prob_rischio"] = y_prob
test_scored["predizione_binaria"] = y_pred

# recupero feature ingegnerizzate dal dataframe originale
engineered_cols = [
    "RECIDIVITA",
    "IS_RT",
    "IS_NR",
    "RT_BUONO",
    "RT_CATTIVO",
    "HAS_STIPENDIO",
    "TREND_SALDO_POS",
    "TREND_DELTA_POS",
    "DELTA_AVERE_DARE_POS",
    "SCORE_RIENTRO_PROXY",
    "PRESSIONE_FINANZIARIA_PROXY",
    "PD_ALTA",
    "FLUSSI_PROTETTIVI",
    "SQUILIBRIO_OPERATIVO"
]

base_engineered = df[[ID_COL] + engineered_cols].drop_duplicates(subset=[ID_COL], keep="first")
test_scored = test_scored.merge(base_engineered, on=ID_COL, how="left")

test_scored["classe_operativa"] = test_scored.apply(
    lambda r: assegna_classe_operativa(
        prob=r["prob_rischio"],
        rt_buono=r["RT_BUONO"],
        pd_alta=r["PD_ALTA"],
        flussi_protettivi=r["FLUSSI_PROTETTIVI"]
    ),
    axis=1
)

# =========================================================
# 12. CROSSTAB TEST
# =========================================================
print("\n" + "=" * 90)
print("Crosstab fasce vs target - TEST SET")
print("=" * 90)

ct = pd.crosstab(test_scored["classe_operativa"], test_scored[TARGET_COL], margins=True)
print(ct)

# tassi per fascia
summary = (
    test_scored
    .groupby("classe_operativa")
    .agg(
        n=(TARGET_COL, "size"),
        prob_media=("prob_rischio", "mean"),
        prob_min=("prob_rischio", "min"),
        prob_max=("prob_rischio", "max"),
        target_rate=(TARGET_COL, "mean")
    )
    .reset_index()
    .sort_values("prob_media", ascending=False)
)

print("\n" + "=" * 90)
print("RIEPILOGO FASCE - TEST SET")
print("=" * 90)
print(summary)

# =========================================================
# 13. SCORING FULL DATASET
# =========================================================
full_prob = model.predict_proba(X)[:, 1]
full_pred = model.predict(X)

scored_df = X.copy()
scored_df[ID_COL] = id_series.values
scored_df[TARGET_COL] = y.values
scored_df["prob_rischio"] = full_prob
scored_df["predizione_binaria"] = full_pred

scored_df = scored_df.merge(base_engineered, on=ID_COL, how="left")

scored_df["classe_operativa"] = scored_df.apply(
    lambda r: assegna_classe_operativa(
        prob=r["prob_rischio"],
        rt_buono=r["RT_BUONO"],
        pd_alta=r["PD_ALTA"],
        flussi_protettivi=r["FLUSSI_PROTETTIVI"]
    ),
    axis=1
)

# =========================================================
# 14. ESTRAZIONE COEFFICIENTI ROBUSTA
# =========================================================
coef_df = pd.DataFrame()

try:
    fitted_preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]

    feature_names = []

    # numeriche
    feature_names.extend(numeric_cols)

    # categoriche
    if len(categorical_cols) > 0:
        cat_pipe = fitted_preprocessor.named_transformers_["cat"]
        fitted_ohe = cat_pipe.named_steps["onehot"]

        try:
            cat_names = fitted_ohe.get_feature_names_out(categorical_cols).tolist()
        except AttributeError:
            cat_names = fitted_ohe.get_feature_names(categorical_cols).tolist()

        feature_names.extend(cat_names)

    coef = clf.coef_.ravel()

    if len(feature_names) == len(coef):
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coef
        })
        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    else:
        print("\nATTENZIONE: mismatch tra numero feature e coefficienti.")
        print(f"Feature names: {len(feature_names)}")
        print(f"Coefficienti: {len(coef)}")

except Exception as e:
    print(f"\nImpossibile estrarre i coefficienti: {e}")

if not coef_df.empty:
    print("\n" + "=" * 90)
    print("TOP COEFFICIENTI MODELLO - ASSOLUTI")
    print("=" * 90)
    print(coef_df.head(25))

    print("\n" + "=" * 90)
    print("TOP COEFFICIENTI POSITIVI (AUMENTANO RISCHIO)")
    print("=" * 90)
    print(coef_df.sort_values("coefficient", ascending=False).head(20))

    print("\n" + "=" * 90)
    print("TOP COEFFICIENTI NEGATIVI (RIDUCONO RISCHIO)")
    print("=" * 90)
    print(coef_df.sort_values("coefficient", ascending=True).head(20))
else:
    print("\nNessun coefficiente estratto.")

# =========================================================
# 15. OUTPUT FILE
# =========================================================
output_scored_path = os.path.join(OUTPUT_DIR, SCORED_FILE)
coef_output_path = os.path.join(OUTPUT_DIR, COEF_FILE)
model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)

# salvataggio excel con piu' sheet
with pd.ExcelWriter(output_scored_path, engine="openpyxl") as writer:
    scored_df.to_excel(writer, sheet_name="scoring_full", index=False)
    test_scored.to_excel(writer, sheet_name="test_scored", index=False)
    summary.to_excel(writer, sheet_name="summary_test", index=False)
    ct.to_excel(writer, sheet_name="crosstab_test")

if not coef_df.empty:
    coef_df.to_excel(coef_output_path, index=False)

# bundle con metadata utili
model_bundle = {
    "model": model,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "engineered_cols": engineered_cols,
    "target_col": TARGET_COL,
    "id_col": ID_COL,
    "threshold_non_toccare": THRESH_NON_TOCCARE,
    "threshold_monitorare": THRESH_MONITORARE,
    "version": "V2_RT_CORRETTA"
}

joblib.dump(model_bundle, model_path)

# =========================================================
# 16. REPORT FINALE
# =========================================================
print("\n" + "=" * 90)
print("SALVATAGGIO COMPLETATO")
print("=" * 90)
print("Output scoring:", output_scored_path)
if not coef_df.empty:
    print("Output coefficienti:", coef_output_path)
print("Modello:", model_path)

print("\n" + "=" * 90)
print("DISTRIBUZIONE CLASSI OPERATIVE - FULL DATASET")
print("=" * 90)
print(scored_df["classe_operativa"].value_counts(dropna=False))

print("\n" + "=" * 90)
print("RECIDIVITA vs TARGET - FULL DATASET")
print("=" * 90)
print(pd.crosstab(df["RECIDIVITA"], df[TARGET_COL], margins=True))

print("\n" + "=" * 90)
print("RT_BUONO vs TARGET - FULL DATASET")
print("=" * 90)
print(pd.crosstab(scored_df["RT_BUONO"], scored_df[TARGET_COL], margins=True))
