# =========================================================
# 0. IMPORT
# =========================================================
import os
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

from IPython.display import display


# =========================================================
# 1. CONFIG
# =========================================================
INPUT_FILE = r"C:\Users\mike_\OneDrive - BNP Paribas\Bureau\PITONE\input.xlsx"
SHEET_NAME = 0
OUTPUT_DIR = r"C:\Users\mike_\OneDrive - BNP Paribas\Bureau\PITONE\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "target"   # <-- cambia se serve
ID_COL = "ndg"          # <-- cambia se serve
RANDOM_STATE = 42
TEST_SIZE = 0.30

# quote portafoglio per fasce operative
# esempio:
# top 20% = NON_TOCCARE_ORA
# successivo 30% = MONITORARE
# resto = INTERVENIRE_PRIMA
SHARE_NON_TOCCARE = 0.20
SHARE_MONITORARE = 0.30

# se hai sentinelle tipo 9999 da trattare come missing
REPLACE_9999_WITH_NAN = True


# =========================================================
# 2. FUNZIONI BASE
# =========================================================
def load_data(input_file, sheet_name=0):
    ext = os.path.splitext(input_file)[1].lower()
    if ext in [".xlsx", ".xls", ".xlsm"]:
        return pd.read_excel(input_file, sheet_name=sheet_name)
    elif ext == ".csv":
        return pd.read_csv(input_file)
    else:
        raise ValueError(f"Formato file non supportato: {ext}")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_target(df, target_col):
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"La colonna target '{target_col}' non esiste nel dataset.")

    df["_target_raw_backup_"] = df[target_col]
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    print("=" * 90)
    print("PULIZIA TARGET")
    print("=" * 90)
    print("Righe iniziali:", len(df))
    print("Target mancanti/non validi:", df[target_col].isna().sum())

    df = df[df[target_col].notna()].copy()
    df[target_col] = df[target_col].astype(int)

    if df.empty:
        raise ValueError("Il dataset è vuoto dopo la pulizia del target.")

    print("Righe finali dopo pulizia:", len(df))
    print("\nDistribuzione target:")
    display(
        df[target_col]
        .value_counts(dropna=False)
        .rename_axis("classe")
        .reset_index(name="conteggio")
    )

    return df


def build_feature_lists(df, target_col, id_col=None):
    exclude_cols = [target_col, "_target_raw_backup_"]
    if id_col is not None and id_col in df.columns:
        exclude_cols.append(id_col)

    feature_columns = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_columns].copy()
    y = df[target_col].copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    return X, y, feature_columns, numeric_cols, categorical_cols


def replace_sentinel_values(df, numeric_cols, sentinel=9999):
    df = df.copy()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace(sentinel, np.nan)
    return df


def build_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    model = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return clf


def extract_model_coefficients(clf):
    try:
        preprocessor = clf.named_steps["preprocessor"]
        model = clf.named_steps["model"]

        coefficients = model.coef_.ravel()

        # Provo prima il recupero diretto
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            # Fallback manuale robusto
            feature_names = []

            for name, transformer, cols in preprocessor.transformers_:
                if transformer == "drop":
                    continue

                if name == "num":
                    feature_names.extend(list(cols))

                elif name == "cat":
                    # categorical_transformer è una Pipeline con step "onehot"
                    ohe = transformer.named_steps["onehot"]
                    cat_feature_names = ohe.get_feature_names_out(cols)
                    feature_names.extend(list(cat_feature_names))

                else:
                    feature_names.extend(list(cols))

        if len(feature_names) != len(coefficients):
            raise ValueError(
                f"Mismatch tra feature names ({len(feature_names)}) "
                f"e coefficienti ({len(coefficients)})"
            )

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients)
        }).sort_values("abs_coefficient", ascending=False)

        return coef_df

    except Exception as e:
        print("Impossibile estrarre i coefficienti:", e)
        return pd.DataFrame()


def compute_dynamic_thresholds(prob_series, share_non_toccare=0.20, share_monitorare=0.30):
    """
    Calcola soglie automatiche in base alle quote di portafoglio.
    """
    if share_non_toccare <= 0 or share_non_toccare >= 1:
        raise ValueError("share_non_toccare deve essere compreso tra 0 e 1")

    if share_monitorare < 0 or share_monitorare >= 1:
        raise ValueError("share_monitorare deve essere compreso tra 0 e 1")

    if share_non_toccare + share_monitorare >= 1:
        raise ValueError("share_non_toccare + share_monitorare deve essere < 1")

    threshold_non_toccare = prob_series.quantile(1 - share_non_toccare)
    threshold_monitorare = prob_series.quantile(1 - (share_non_toccare + share_monitorare))

    return float(threshold_non_toccare), float(threshold_monitorare)


def assign_operational_band(prob, threshold_non_toccare, threshold_monitorare):
    if prob >= threshold_non_toccare:
        return "NON_TOCCARE_ORA"
    elif prob >= threshold_monitorare:
        return "MONITORARE"
    else:
        return "INTERVENIRE_PRIMA"


def add_operational_columns(df, prob_col, threshold_non_toccare, threshold_monitorare):
    df = df.copy()

    df["classe_operativa"] = df[prob_col].apply(
        lambda x: assign_operational_band(
            x,
            threshold_non_toccare=threshold_non_toccare,
            threshold_monitorare=threshold_monitorare
        )
    )

    # ranking interno
    df["rank_priorita"] = df[prob_col].rank(ascending=False, method="dense").astype(int)

    return df


def evaluate_binary_cut(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    print("=" * 90)
    print(f"VALUTAZIONE BINARIA - threshold = {threshold:.4f}")
    print("=" * 90)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_true, y_proba)
        print(f"ROC AUC: {auc:.4f}")
    except Exception as e:
        print("ROC AUC non calcolabile:", e)


def summarize_operational_bands(df, target_col=None):
    if target_col is not None and target_col in df.columns:
        summary = (
            df.groupby("classe_operativa")
              .agg(
                  n_clienti=("classe_operativa", "size"),
                  prob_min=("prob_tappeto_1", "min"),
                  prob_media=("prob_tappeto_1", "mean"),
                  prob_max=("prob_tappeto_1", "max"),
                  target_rate=(target_col, "mean")
              )
              .sort_values("prob_media", ascending=False)
              .reset_index()
        )
    else:
        summary = (
            df.groupby("classe_operativa")
              .agg(
                  n_clienti=("classe_operativa", "size"),
                  prob_min=("prob_tappeto_1", "min"),
                  prob_media=("prob_tappeto_1", "mean"),
                  prob_max=("prob_tappeto_1", "max")
              )
              .sort_values("prob_media", ascending=False)
              .reset_index()
        )

    return summary


# =========================================================
# 3. LETTURA DATI
# =========================================================
df = load_data(INPUT_FILE, SHEET_NAME)
df = clean_columns(df)

print("=" * 90)
print("DATASET CARICATO")
print("=" * 90)
print("Shape iniziale:", df.shape)
display(df.head())


# =========================================================
# 4. PULIZIA TARGET
# =========================================================
df = clean_target(df, TARGET_COL)


# =========================================================
# 5. FEATURE
# =========================================================
X, y, feature_columns, numeric_cols, categorical_cols = build_feature_lists(
    df, TARGET_COL, ID_COL
)

if REPLACE_9999_WITH_NAN and len(numeric_cols) > 0:
    X = replace_sentinel_values(X, numeric_cols, sentinel=9999)

print("=" * 90)
print("FEATURE SET")
print("=" * 90)
print("Numero feature originali:", len(feature_columns))
print("Numero feature numeriche:", len(numeric_cols))
print("Numero feature categoriche:", len(categorical_cols))


# =========================================================
# 6. TRAIN / TEST
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("=" * 90)
print("TRAIN / TEST")
print("=" * 90)
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# =========================================================
# 7. MODELLO
# =========================================================
clf = build_pipeline(numeric_cols, categorical_cols)
clf.fit(X_train, y_train)

y_proba_test = clf.predict_proba(X_test)[:, 1]

try:
    auc = roc_auc_score(y_test, y_proba_test)
    print(f"\nROC AUC TEST: {auc:.4f}")
except Exception as e:
    print("ROC AUC non calcolabile:", e)


# =========================================================
# 8. SOGLIE DINAMICHE SUL TEST SET
# =========================================================
threshold_non_toccare, threshold_monitorare = compute_dynamic_thresholds(
    pd.Series(y_proba_test),
    share_non_toccare=SHARE_NON_TOCCARE,
    share_monitorare=SHARE_MONITORARE
)

print("=" * 90)
print("SOGLIE DINAMICHE CALCOLATE SUL TEST SET")
print("=" * 90)
print(f"Quota NON_TOCCARE_ORA : {SHARE_NON_TOCCARE:.0%}")
print(f"Quota MONITORARE      : {SHARE_MONITORARE:.0%}")
print(f"Soglia NON_TOCCARE    : {threshold_non_toccare:.6f}")
print(f"Soglia MONITORARE     : {threshold_monitorare:.6f}")

print("\n")
evaluate_binary_cut(y_test, y_proba_test, threshold_monitorare)


# =========================================================
# 9. ANALISI FASCE SUL TEST SET
# =========================================================
test_scored = X_test.copy()
test_scored[TARGET_COL] = y_test.values
test_scored["prob_tappeto_1"] = y_proba_test

test_scored = add_operational_columns(
    test_scored,
    prob_col="prob_tappeto_1",
    threshold_non_toccare=threshold_non_toccare,
    threshold_monitorare=threshold_monitorare
)

print("=" * 90)
print("DISTRIBUZIONE FASCE - TEST SET")
print("=" * 90)
display(
    test_scored["classe_operativa"]
    .value_counts(dropna=False)
    .rename_axis("classe_operativa")
    .reset_index(name="conteggio")
)

print("\nAnalisi fasce vs target - TEST SET")
display(summarize_operational_bands(test_scored, TARGET_COL))

print("\nCrosstab fasce vs target - TEST SET")
display(pd.crosstab(test_scored["classe_operativa"], test_scored[TARGET_COL], margins=True))


# =========================================================
# 10. COEFFICIENTI MODELLO
# =========================================================
coef_df = extract_model_coefficients(clf)

if not coef_df.empty:
    print("=" * 90)
    print("TOP COEFFICIENTI MODELLO - ASSOLUTI")
    print("=" * 90)
    display(coef_df.head(30))

    print("=" * 90)
    print("TOP COEFFICIENTI POSITIVI")
    print("=" * 90)
    display(coef_df.sort_values("coefficient", ascending=False).head(20))

    print("=" * 90)
    print("TOP COEFFICIENTI NEGATIVI")
    print("=" * 90)
    display(coef_df.sort_values("coefficient", ascending=True).head(20))
else:
    print("Nessun coefficiente estratto.")


# =========================================================
# 11. SCORING SU TUTTO IL DATASET
# =========================================================
df_scored = df.copy()

X_full = df[feature_columns].copy()
if REPLACE_9999_WITH_NAN and len(numeric_cols) > 0:
    X_full = replace_sentinel_values(X_full, numeric_cols, sentinel=9999)

df_scored["prob_tappeto_1"] = clf.predict_proba(X_full)[:, 1]

# puoi scegliere:
# A) usare le soglie calcolate sul test set
# B) ricalcolare le quote sul full dataset
# sotto uso B, più coerente con la capacità operativa finale
threshold_non_toccare_full, threshold_monitorare_full = compute_dynamic_thresholds(
    df_scored["prob_tappeto_1"],
    share_non_toccare=SHARE_NON_TOCCARE,
    share_monitorare=SHARE_MONITORARE
)

df_scored = add_operational_columns(
    df_scored,
    prob_col="prob_tappeto_1",
    threshold_non_toccare=threshold_non_toccare_full,
    threshold_monitorare=threshold_monitorare_full
)

df_scored = df_scored.sort_values("prob_tappeto_1", ascending=False).reset_index(drop=True)

print("=" * 90)
print("SOGLIE DINAMICHE - FULL DATASET")
print("=" * 90)
print(f"Soglia NON_TOCCARE full: {threshold_non_toccare_full:.6f}")
print(f"Soglia MONITORARE full : {threshold_monitorare_full:.6f}")

print("\nDistribuzione classi operative - FULL DATASET")
display(
    df_scored["classe_operativa"]
    .value_counts(dropna=False)
    .rename_axis("classe_operativa")
    .reset_index(name="conteggio")
)

print("\nAnalisi fasce vs target - FULL DATASET")
display(summarize_operational_bands(df_scored, TARGET_COL))

print("\nCrosstab fasce vs target - FULL DATASET")
display(pd.crosstab(df_scored["classe_operativa"], df_scored[TARGET_COL], margins=True))

print("\nTop clienti per probabilità:")
display(df_scored.head(20))


# =========================================================
# 12. SALVATAGGIO
# =========================================================
output_scored_path = os.path.join(OUTPUT_DIR, "dataset_scored_operativo_v2.xlsx")
df_scored.to_excel(output_scored_path, index=False)

# salvo anche i coefficienti, se disponibili
coef_output_path = os.path.join(OUTPUT_DIR, "coefficienti_modello_v2.xlsx")
if not coef_df.empty:
    coef_df.to_excel(coef_output_path, index=False)

model_bundle = {
    "pipeline": clf,
    "feature_columns": feature_columns,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "target_col": TARGET_COL,
    "id_col": ID_COL,
    "share_non_toccare": SHARE_NON_TOCCARE,
    "share_monitorare": SHARE_MONITORARE,
    "threshold_non_toccare_test": threshold_non_toccare,
    "threshold_monitorare_test": threshold_monitorare,
    "threshold_non_toccare_full": threshold_non_toccare_full,
    "threshold_monitorare_full": threshold_monitorare_full,
    "replace_9999_with_nan": REPLACE_9999_WITH_NAN
}

model_path = os.path.join(OUTPUT_DIR, "modello_cluster_tappeto_operativo_v2.joblib")
joblib.dump(model_bundle, model_path)

print("=" * 90)
print("SALVATAGGIO COMPLETATO")
print("=" * 90)
print("Output scoring:", output_scored_path)
if not coef_df.empty:
    print("Output coefficienti:", coef_output_path)
print("Modello:", model_path)
