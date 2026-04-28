# -*- coding: utf-8 -*-

# ============================================================
# CLUSTER TAPPETO / SELF-CURE MODEL
# VERSIONE OTTIMIZZATA COMPLETA
# XGBOOST + FEATURE ENGINEERING + THRESHOLD + ROI + SCORING
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r"C:\Users\D75737\OneDrive - BNP Paribas\Bureau\cl4.xlsx"
SHEET_NAME = 0

TARGET_COL = "Cluster"     # 1 = autoregolarizza / tappeto
ID_COL = "ndg"

OUTPUT_DIR = r"output_tappeto_xgb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE = 0.30
RANDOM_STATE = 42

# ROI
COSTO_CONTATTO = 5.0
RECUPERO_MEDIO = 200.0

# scelta threshold finale:
# "f1" = bilancia precision/recall classe 1
# "roi" = massimizza profitto simulato
THRESHOLD_MODE = "f1"


# ============================================================
# COLONNE DA ESCLUDERE
# ============================================================

LEAKAGE_COLS = [
    ID_COL,
    "RATING_MINORE",
    "FASCIA_RISCHIO",
    "COD_FASCIA_RISCHIO",
    "FLAG_RECIDIVO",
    "S_TREND_STATUS",
    "NUM_GG_DA_USCITA",
    "GG_IRREGOLARE",
    "GG_ULTIMA_AZIONE",
    "DT_CONT",
    "Dt cont",
    "DT_CONT_MESE",
    "DT_CONT_GIORNO",
    "DT_CONT_DAYOFWEEK"
]

BORDERLINE_COLS = [
    # "RECIDIVITA",
    # "R_PD_ULT_MENO_3MESI",
    # "MESE",
]

SENTINEL_VALUES = [9999, -9999, 999999, -999999]


# ============================================================
# COLONNE ATTESE
# ============================================================

COL_STIP_AVG_3M = "CC_IMP_STIPEND_AVG_3M"
COL_SCONF_TOT = "IMP_SCONFINMTO_TOT_UM"
COL_SALDO_AVG_3M = "CC_SALDO_AVG_3M"
COL_CONTANTI_AVG_3M = "CC_IMP_CONTANTI_AVG_3M"
COL_UTENZE_AVG_3M = "CC_IMP_UTENZE_AVG_3M"
COL_UTILZ_CF = "F_PER_UTILZ_CF_UM"
COL_NUM_STIP_AVG_3M = "CC_NUM_STIPENDIO_AVG_3M"
COL_TREND_STIP_3_12M = "CC_TREND_STIPEND_3_12M"
COL_PD_ULT_3M = "R_PD_ULT_MENO_3MESI"


# ============================================================
# FUNZIONI BASE
# ============================================================

def load_data(file_path, sheet_name=0):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".xlsx", ".xls", ".xlsm"]:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    elif ext == ".csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Formato file non supportato: {ext}")


def standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_div(a, b):
    return a / (b + 1e-6)


def add_feature_if_possible(df, new_col, func, required_cols):
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"[INFO] Feature non creata: {new_col} | colonne mancanti: {missing}")
        return df

    try:
        df[new_col] = func(df)
    except Exception as e:
        print(f"[WARN] impossibile creare {new_col}: {e}")

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].astype(str).str.strip()

            s2 = (
                s.str.replace(".", "", regex=False)
                 .str.replace(",", ".", regex=False)
                 .str.replace(" ", "", regex=False)
            )

            converted = pd.to_numeric(s2, errors="coerce")

            if converted.notna().mean() >= 0.70:
                df[col] = converted

    df = df.replace(SENTINEL_VALUES, np.nan)

    df = add_feature_if_possible(
        df,
        "CAPACITA_RIENTRO",
        lambda x: x[COL_STIP_AVG_3M] - x[COL_SCONF_TOT],
        [COL_STIP_AVG_3M, COL_SCONF_TOT]
    )

    df = add_feature_if_possible(
        df,
        "STRESS_RATIO",
        lambda x: safe_div(x[COL_SCONF_TOT], x[COL_STIP_AVG_3M] + 1),
        [COL_STIP_AVG_3M, COL_SCONF_TOT]
    )

    df = add_feature_if_possible(
        df,
        "LIQUIDITA_NETTA",
        lambda x: x[COL_SALDO_AVG_3M] - x[COL_SCONF_TOT],
        [COL_SALDO_AVG_3M, COL_SCONF_TOT]
    )

    df = add_feature_if_possible(
        df,
        "ATTIVITA_3M",
        lambda x: x[COL_CONTANTI_AVG_3M] + x[COL_UTENZE_AVG_3M],
        [COL_CONTANTI_AVG_3M, COL_UTENZE_AVG_3M]
    )

    df = add_feature_if_possible(
        df,
        "LEVA_CF",
        lambda x: x[COL_UTILZ_CF],
        [COL_UTILZ_CF]
    )

    df = add_feature_if_possible(
        df,
        "SALARY_REGULARITY",
        lambda x: safe_div(x[COL_NUM_STIP_AVG_3M], 3),
        [COL_NUM_STIP_AVG_3M]
    )

    df = add_feature_if_possible(
        df,
        "CAPACITA_RIENTRO_RELATIVA",
        lambda x: safe_div(x[COL_STIP_AVG_3M], x[COL_SCONF_TOT] + 1),
        [COL_STIP_AVG_3M, COL_SCONF_TOT]
    )

    df = add_feature_if_possible(
        df,
        "ATTIVITA_SU_STIPENDIO",
        lambda x: safe_div(
            x[COL_CONTANTI_AVG_3M] + x[COL_UTENZE_AVG_3M],
            x[COL_STIP_AVG_3M] + 1
        ),
        [COL_CONTANTI_AVG_3M, COL_UTENZE_AVG_3M, COL_STIP_AVG_3M]
    )

    df = add_feature_if_possible(
        df,
        "SALDO_SU_SCONF",
        lambda x: safe_div(x[COL_SALDO_AVG_3M], x[COL_SCONF_TOT] + 1),
        [COL_SALDO_AVG_3M, COL_SCONF_TOT]
    )

    df = add_feature_if_possible(
        df,
        "TREND_STIPEND_POSITIVO",
        lambda x: (x[COL_TREND_STIP_3_12M].fillna(0) > 0).astype(int),
        [COL_TREND_STIP_3_12M]
    )

    df = add_feature_if_possible(
        df,
        "RECIDIVITA_BREVE_FLAG",
        lambda x: (x[COL_PD_ULT_3M].fillna(0) > 0).astype(int),
        [COL_PD_ULT_3M]
    )

    return df


# ============================================================
# SEGMENTAZIONE OPERATIVA
# ============================================================

def build_segments(prob):
    if prob >= 0.70:
        return "AUTO"
    elif prob >= 0.40:
        return "SOFT"
    else:
        return "HARD"


def build_action(segmento):
    if segmento == "AUTO":
        return "NON_CONTATTARE"
    elif segmento == "SOFT":
        return "CONTATTO_SOFT"
    else:
        return "CONTATTO_PRIORITARIO"


# ============================================================
# REPORT EXCEL
# ============================================================

def save_excel_report(path, sheets_dict):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets_dict.items():
            if df_sheet is None:
                continue

            if not isinstance(df_sheet, pd.DataFrame):
                df_sheet = pd.DataFrame(df_sheet)

            df_sheet.to_excel(
                writer,
                sheet_name=str(sheet_name)[:31],
                index=False
            )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance_from_pipeline(clf):
    try:
        preprocessor = clf.named_steps["preprocessor"]
        model = clf.named_steps["model"]

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

        importances = model.feature_importances_

        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return feat_imp

    except Exception as e:
        print(f"[WARN] impossibile estrarre feature importance: {e}")
        return pd.DataFrame(columns=["feature", "importance"])


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_best_threshold_f1(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    rows = []

    for i, t in enumerate(thresholds):
        p = precision[i]
        r = recall[i]
        f1 = 2 * p * r / (p + r + 1e-9)

        rows.append({
            "threshold": float(t),
            "precision_1": float(p),
            "recall_1": float(r),
            "f1_1": float(f1)
        })

    df_thr = pd.DataFrame(rows).sort_values("f1_1", ascending=False).reset_index(drop=True)
    best_threshold = float(df_thr.iloc[0]["threshold"])

    return best_threshold, df_thr


def evaluate_thresholds_roi(y_true, y_proba, cost_contact=5.0, avg_recovery=200.0):
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_proba = pd.Series(y_proba).reset_index(drop=True)

    for t in thresholds:
        pred_tappeto = (y_proba >= t).astype(int)

        # se pred_tappeto = 1 => AUTO / non contattare
        contact_flag = 1 - pred_tappeto

        contacts = int(contact_flag.sum())

        recovered_non_tappeto = int(((contact_flag == 1) & (y_true == 0)).sum())

        cost = contacts * cost_contact
        revenue = recovered_non_tappeto * avg_recovery
        profit = revenue - cost

        rows.append({
            "threshold": float(t),
            "contacts": contacts,
            "recovered_non_tappeto": recovered_non_tappeto,
            "cost": cost,
            "revenue": revenue,
            "profit": profit
        })

    return pd.DataFrame(rows).sort_values("profit", ascending=False).reset_index(drop=True)


# ============================================================
# LIFT TABLE
# ============================================================

def build_lift_table(y_true, y_proba):
    df_lift = pd.DataFrame({
        "y_true": pd.Series(y_true).reset_index(drop=True),
        "prob_tappeto": pd.Series(y_proba).reset_index(drop=True)
    }).sort_values("prob_tappeto", ascending=False).reset_index(drop=True)

    df_lift["rank"] = np.arange(1, len(df_lift) + 1)
    df_lift["perc"] = df_lift["rank"] / len(df_lift)

    total_positive = df_lift["y_true"].sum()
    df_lift["cum_positive"] = df_lift["y_true"].cumsum()

    df_lift["lift"] = np.where(
        df_lift["perc"] > 0,
        df_lift["cum_positive"] / (df_lift["perc"] * total_positive + 1e-9),
        np.nan
    )

    return df_lift


# ============================================================
# SCORING NUOVO FILE
# ============================================================

def score_new_file(model_path, new_file_path, output_file_path, sheet_name=0, threshold=None):
    clf = joblib.load(model_path)

    if threshold is None:
        threshold = clf.get("threshold", 0.50) if isinstance(clf, dict) else 0.50

    if isinstance(clf, dict):
        model_pipeline = clf["model"]
    else:
        model_pipeline = clf

    new_df = load_data(new_file_path, sheet_name=sheet_name)
    new_df = standardize_columns(new_df)
    new_df = feature_engineering(new_df)

    drop_cols = [c for c in [TARGET_COL, ID_COL] if c in new_df.columns]
    X_new = new_df.drop(columns=drop_cols, errors="ignore")

    new_df["prob_tappeto"] = model_pipeline.predict_proba(X_new)[:, 1]
    new_df["pred_tappeto"] = (new_df["prob_tappeto"] >= threshold).astype(int)
    new_df["segmento"] = new_df["prob_tappeto"].apply(build_segments)
    new_df["azione_suggerita"] = new_df["segmento"].apply(build_action)

    new_df = new_df.sort_values("prob_tappeto", ascending=False).reset_index(drop=True)
    new_df["rank"] = np.arange(1, len(new_df) + 1)
    new_df["percentile_rank"] = new_df["rank"] / len(new_df)

    new_df.to_excel(output_file_path, index=False)

    print("\nScoring completato.")
    print(f"Threshold usata: {threshold:.4f}")
    print(f"File salvato in: {output_file_path}")

    display(new_df.head(20))

    return new_df


# ============================================================
# 1. LETTURA DATI
# ============================================================

df = load_data(INPUT_FILE, SHEET_NAME)
df = standardize_columns(df)

print("=" * 90)
print("DATASET CARICATO")
print("=" * 90)
print("Shape iniziale:", df.shape)
display(df.head())


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

df = feature_engineering(df)

print("\nShape dopo feature engineering:", df.shape)


# ============================================================
# 3. TARGET CLEAN
# ============================================================

if TARGET_COL not in df.columns:
    raise ValueError(f"La colonna target '{TARGET_COL}' non esiste nel dataset.")

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df[df[TARGET_COL].notna()].copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

if df.empty:
    raise ValueError("Il dataset è vuoto dopo la pulizia del target.")

if df[TARGET_COL].nunique() < 2:
    raise ValueError("Il target ha una sola classe: impossibile allenare il modello.")

print("\nDistribuzione target:")
target_dist = (
    df[TARGET_COL]
    .value_counts(dropna=False)
    .rename_axis("classe")
    .reset_index(name="conteggio")
)
display(target_dist)


# ============================================================
# 4. DROP COLONNE SPORCHE / LEAKAGE
# ============================================================

drop_cols = [TARGET_COL]

if ID_COL in df.columns:
    drop_cols.append(ID_COL)

drop_cols += [c for c in LEAKAGE_COLS if c in df.columns]
drop_cols += [c for c in BORDERLINE_COLS if c in df.columns]

drop_cols = list(dict.fromkeys(drop_cols))

X = df.drop(columns=drop_cols, errors="ignore").copy()
y = df[TARGET_COL].copy()

# rimuovi datetime grezze
date_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
if len(date_cols) > 0:
    print("Rimuovo colonne datetime:", date_cols)
    X = X.drop(columns=date_cols, errors="ignore")

# elimina colonne completamente vuote
X = X.loc[:, X.notna().sum() > 0].copy()

num_cols = X.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumero feature candidate:", X.shape[1])
print("Feature numeriche:", len(num_cols))
print("Feature categoriche:", len(cat_cols))

print("\nPrime feature numeriche:")
print(num_cols[:20])

print("\nPrime feature categoriche:")
print(cat_cols[:20])


# ============================================================
# 5. PROFILO DESCRITTIVO
# ============================================================

df_pos = df[df[TARGET_COL] == 1].copy()
df_neg = df[df[TARGET_COL] == 0].copy()

numeric_profile_rows = []

for col in num_cols:
    try:
        pos_mean = df_pos[col].mean()
        neg_mean = df_neg[col].mean()
        pos_median = df_pos[col].median()
        neg_median = df_neg[col].median()
        diff_mean = pos_mean - neg_mean

        numeric_profile_rows.append({
            "variabile": col,
            "mean_tappeto_1": pos_mean,
            "median_tappeto_1": pos_median,
            "mean_tappeto_0": neg_mean,
            "median_tappeto_0": neg_median,
            "diff_mean": diff_mean,
            "abs_diff_mean": abs(diff_mean) if pd.notna(diff_mean) else np.nan
        })
    except Exception:
        pass

numeric_profile_df = pd.DataFrame(numeric_profile_rows)

if not numeric_profile_df.empty:
    numeric_profile_df = numeric_profile_df.sort_values(
        "abs_diff_mean",
        ascending=False
    ).reset_index(drop=True)

print("\nTop differenze variabili numeriche:")
display(numeric_profile_df.head(20))


categorical_profile_all = []

for col in cat_cols:
    try:
        tmp = (
            df.groupby([col, TARGET_COL], dropna=False)
            .size()
            .reset_index(name="n")
        )

        pivot = tmp.pivot_table(
            index=col,
            columns=TARGET_COL,
            values="n",
            fill_value=0
        )

        pivot.columns = [f"count_target_{int(c)}" for c in pivot.columns]
        pivot = pivot.reset_index()

        if "count_target_1" not in pivot.columns:
            pivot["count_target_1"] = 0

        if "count_target_0" not in pivot.columns:
            pivot["count_target_0"] = 0

        pivot["total"] = pivot["count_target_1"] + pivot["count_target_0"]

        pivot["pct_tappeto_1_nel_livello"] = np.where(
            pivot["total"] > 0,
            pivot["count_target_1"] / pivot["total"],
            np.nan
        )

        pivot["variabile"] = col

        pivot = pivot.sort_values(
            ["pct_tappeto_1_nel_livello", "count_target_1"],
            ascending=[False, False]
        )

        categorical_profile_all.append(pivot.head(10))

    except Exception:
        pass

categorical_profile_df = (
    pd.concat(categorical_profile_all, ignore_index=True)
    if categorical_profile_all else pd.DataFrame()
)

print("\nTop livelli categorici associati a tappeto=1:")
display(categorical_profile_df.head(40))


# ============================================================
# 6. PREPROCESSING
# ============================================================

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
    ],
    remainder="drop"
)


# ============================================================
# 7. TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

print("\nScale_pos_weight:", scale_pos_weight)


# ============================================================
# 8. MODELLO XGBOOST OTTIMIZZATO
# ============================================================

model = XGBClassifier(
    n_estimators=700,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.10,
    reg_lambda=1.00,
    reg_alpha=0.10,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

clf.fit(X_train, y_train)


# ============================================================
# 9. EVALUATION + THRESHOLD
# ============================================================

y_proba = clf.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)

best_threshold_f1, threshold_table = find_best_threshold_f1(y_test, y_proba)

roi_df = evaluate_thresholds_roi(
    y_true=y_test.reset_index(drop=True),
    y_proba=y_proba,
    cost_contact=COSTO_CONTATTO,
    avg_recovery=RECUPERO_MEDIO
)

best_threshold_roi = float(roi_df.iloc[0]["threshold"])

if THRESHOLD_MODE.lower() == "roi":
    FINAL_THRESHOLD = best_threshold_roi
else:
    FINAL_THRESHOLD = best_threshold_f1

y_pred = (y_proba >= FINAL_THRESHOLD).astype(int)

print("\n" + "=" * 90)
print("VALUTAZIONE MODELLO")
print("=" * 90)
print(f"ROC AUC: {auc:.4f}")
print(f"Best threshold F1:  {best_threshold_f1:.4f}")
print(f"Best threshold ROI: {best_threshold_roi:.4f}")
print(f"Threshold finale usata: {FINAL_THRESHOLD:.4f}")
print(f"Modalità threshold: {THRESHOLD_MODE}")

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, digits=4))

print("\nMetriche classe 1:")
print("Precision classe 1:", precision_score(y_test, y_pred))
print("Recall classe 1:", recall_score(y_test, y_pred))
print("F1 classe 1:", f1_score(y_test, y_pred))


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

feat_imp = compute_feature_importance_from_pipeline(clf)

print("\n=== TOP FEATURE ===")
display(feat_imp.head(30))


# ============================================================
# 11. SCORING COMPLETO DATASET
# ============================================================

df_scored = df.copy()
df_scored["prob_tappeto"] = clf.predict_proba(X)[:, 1]
df_scored["pred_tappeto"] = (df_scored["prob_tappeto"] >= FINAL_THRESHOLD).astype(int)
df_scored["segmento"] = df_scored["prob_tappeto"].apply(build_segments)
df_scored["azione_suggerita"] = df_scored["segmento"].apply(build_action)

df_scored = df_scored.sort_values("prob_tappeto", ascending=False).reset_index(drop=True)
df_scored["rank"] = np.arange(1, len(df_scored) + 1)
df_scored["percentile_rank"] = df_scored["rank"] / len(df_scored)

top_cols = [
    c for c in [
        ID_COL,
        "prob_tappeto",
        "pred_tappeto",
        "segmento",
        "azione_suggerita",
        TARGET_COL
    ]
    if c in df_scored.columns
]

top_clients = df_scored[top_cols].sort_values(
    "prob_tappeto",
    ascending=False
).reset_index(drop=True)

print("\n=== TOP CLIENTI ===")
display(top_clients.head(20))


# ============================================================
# 12. LIFT TABLE
# ============================================================

lift_df = build_lift_table(y_test.reset_index(drop=True), pd.Series(y_proba))

print("\n=== LIFT TABLE ===")
display(lift_df[["perc", "lift"]].head(20))

plt.figure(figsize=(8, 5))
plt.plot(lift_df["perc"], lift_df["lift"])
plt.xlabel("Percentuale clienti selezionati")
plt.ylabel("Lift")
plt.title("Lift Curve")
plt.grid(True)
plt.show()


# ============================================================
# 13. ROI TABLE
# ============================================================

print("\n=== TOP SOGLIE ROI ===")
display(roi_df.head(10))

plt.figure(figsize=(8, 5))
plt.plot(roi_df["threshold"], roi_df["profit"])
plt.xlabel("Threshold")
plt.ylabel("Profitto simulato")
plt.title("Profitto vs Threshold")
plt.grid(True)
plt.show()


# ============================================================
# 14. EXPORT FILE
# ============================================================

feature_path = os.path.join(OUTPUT_DIR, "feature_importance_tappeto.xlsx")
profile_num_path = os.path.join(OUTPUT_DIR, "profilo_numerico_tappeto.xlsx")
profile_cat_path = os.path.join(OUTPUT_DIR, "profilo_categorico_tappeto.xlsx")
scored_path = os.path.join(OUTPUT_DIR, "dataset_scored_tappeto.xlsx")
top_path = os.path.join(OUTPUT_DIR, "top_clienti_prob_tappeto.xlsx")
report_path = os.path.join(OUTPUT_DIR, "report_tappeto_xgb.xlsx")
model_path = os.path.join(OUTPUT_DIR, "modello_tappeto_xgb.joblib")

feat_imp.to_excel(feature_path, index=False)
numeric_profile_df.to_excel(profile_num_path, index=False)
categorical_profile_df.to_excel(profile_cat_path, index=False)
df_scored.to_excel(scored_path, index=False)
top_clients.to_excel(top_path, index=False)

summary_df = pd.DataFrame([{
    "roc_auc": auc,
    "best_threshold_f1": best_threshold_f1,
    "best_threshold_roi": best_threshold_roi,
    "final_threshold": FINAL_THRESHOLD,
    "threshold_mode": THRESHOLD_MODE,
    "precision_1": precision_score(y_test, y_pred),
    "recall_1": recall_score(y_test, y_pred),
    "f1_1": f1_score(y_test, y_pred),
    "n_rows": len(df),
    "n_features": X.shape[1],
    "n_numeric_features": len(num_cols),
    "n_categorical_features": len(cat_cols),
    "scale_pos_weight": scale_pos_weight
}])

save_excel_report(
    report_path,
    {
        "summary": summary_df,
        "threshold_f1_table": threshold_table,
        "roi_table": roi_df,
        "feature_importance": feat_imp,
        "profilo_numerico": numeric_profile_df,
        "profilo_categorico": categorical_profile_df,
        "top_clienti": top_clients.head(500),
        "lift_table": lift_df
    }
)

model_bundle = {
    "model": clf,
    "threshold": FINAL_THRESHOLD,
    "threshold_mode": THRESHOLD_MODE,
    "target_col": TARGET_COL,
    "id_col": ID_COL,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "drop_cols": drop_cols
}

joblib.dump(model_bundle, model_path)

print("\n" + "=" * 90)
print("FILE SALVATI")
print("=" * 90)
print("Feature importance:", feature_path)
print("Profilo numerico:", profile_num_path)
print("Profilo categorico:", profile_cat_path)
print("Dataset scored:", scored_path)
print("Top clienti:", top_path)
print("Report Excel:", report_path)
print("Modello salvato:", model_path)


# ============================================================
# 15. ESEMPIO RIUTILIZZO SU NUOVO FILE
# ============================================================

# Decommenta quando vuoi applicare il modello a una nuova lista:
#
# score_new_file(
#     model_path=model_path,
#     new_file_path=r"C:\Users\YOUR_USER\Desktop\nuova_lista_ndg.xlsx",
#     output_file_path=os.path.join(OUTPUT_DIR, "scoring_nuova_lista.xlsx"),
#     sheet_name=0
# )
    
