# ============================================================
# CLUSTER TAPPETO - ANALISI PROFILO CLIENTI + MODELLO SALVABILE
# Versione ottimizzata
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

# ============================================================
# PARAMETRI UTENTE
# ============================================================

INPUT_FILE = r"C:\percorso\tuo_file.xlsx"   # <-- cambia qui
SHEET_NAME = 0                              # oppure nome foglio
TARGET_COL = "tappeto"                      # <-- variabile target
ID_COL = None                               # es. "ndg" se vuoi escluderla dalle feature
OUTPUT_DIR = r"C:\percorso\output_tappeto"  # <-- cambia qui

# colonne da escludere a priori se presenti
EXCLUDE_COLS = [
    TARGET_COL
]

if ID_COL is not None:
    EXCLUDE_COLS.append(ID_COL)

# soglie utili
MIN_UNIQUE_FOR_NUMERIC = 15     # se una colonna numerica ha pochissimi valori distinti, può essere trattata come categorica
TOP_N_CATEGORY_LEVELS = 15      # per output profilo categorie
TEST_SIZE = 0.30
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LETTURA DATI
# ============================================================

def load_data(file_path, sheet_name=0):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xlsm", ".xls"]:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Formato file non supportato: {ext}")
    return df

df = load_data(INPUT_FILE, SHEET_NAME)

print("=" * 80)
print("DATASET CARICATO")
print("=" * 80)
print(f"Shape iniziale: {df.shape}")
print(df.head())

# ============================================================
# CONTROLLI PRELIMINARI
# ============================================================

if TARGET_COL not in df.columns:
    raise ValueError(f"La colonna target '{TARGET_COL}' non esiste nel dataset.")

# pulizia nomi colonne
df.columns = [str(c).strip() for c in df.columns]

# target pulito
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

# teniamo solo righe con target valorizzato
df = df[df[TARGET_COL].notna()].copy()

# forziamo target intero
df[TARGET_COL] = df[TARGET_COL].astype(int)

print("\nDistribuzione target:")
print(df[TARGET_COL].value_counts(dropna=False))

if df.empty:
    raise ValueError("Il dataset è vuoto dopo la pulizia del target.")

# ============================================================
# COSTRUZIONE FEATURE SET
# ============================================================

candidate_features = [c for c in df.columns if c not in EXCLUDE_COLS]

# rimuovi colonne completamente vuote
candidate_features = [c for c in candidate_features if df[c].notna().sum() > 0]

X = df[candidate_features].copy()
y = df[TARGET_COL].copy()

# prova conversione numerica leggera su object che sembrano numeriche
for col in X.columns:
    if X[col].dtype == "object":
        tmp = pd.to_numeric(X[col].astype(str).str.replace(",", "."), errors="coerce")
        # converto solo se almeno il 70% dei valori è interpretabile come numero
        if tmp.notna().mean() >= 0.70:
            X[col] = tmp

# riconoscimento colonne numeriche e categoriche
numeric_cols = []
categorical_cols = []

for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        # se numerica ma con pochi valori distinti, può essere più sensata come categoria
        nunique = X[col].nunique(dropna=True)
        if nunique <= MIN_UNIQUE_FOR_NUMERIC:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    else:
        categorical_cols.append(col)

print("\nColonne numeriche:", numeric_cols)
print("\nColonne categoriche:", categorical_cols)

# ============================================================
# ANALISI DESCRITTIVA DEL PROFILO TAPPETO = 1
# ============================================================

df_pos = df[df[TARGET_COL] == 1].copy()
df_neg = df[df[TARGET_COL] == 0].copy()

print("\n" + "=" * 80)
print("PROFILO DESCRITTIVO CLIENTI CON TAPPETO = 1")
print("=" * 80)
print(f"Numero clienti tappeto=1: {len(df_pos)}")
print(f"Numero clienti tappeto=0: {len(df_neg)}")

# ---- Statistiche numeriche
numeric_profile_rows = []

for col in numeric_cols:
    pos_mean = df_pos[col].mean()
    neg_mean = df_neg[col].mean() if len(df_neg) > 0 else np.nan
    pos_median = df_pos[col].median()
    neg_median = df_neg[col].median() if len(df_neg) > 0 else np.nan
    diff_mean = pos_mean - neg_mean if pd.notna(neg_mean) else np.nan

    numeric_profile_rows.append({
        "variabile": col,
        "mean_tappeto_1": pos_mean,
        "median_tappeto_1": pos_median,
        "mean_tappeto_0": neg_mean,
        "median_tappeto_0": neg_median,
        "diff_mean": diff_mean
    })

numeric_profile_df = pd.DataFrame(numeric_profile_rows)

if not numeric_profile_df.empty:
    numeric_profile_df["abs_diff_mean"] = numeric_profile_df["diff_mean"].abs()
    numeric_profile_df = numeric_profile_df.sort_values("abs_diff_mean", ascending=False)
    print("\nTop differenze variabili numeriche:")
    print(numeric_profile_df.head(20).drop(columns=["abs_diff_mean"]))
else:
    print("\nNessuna variabile numerica utile trovata.")

# ---- Statistiche categoriche
categorical_profile_all = []

for col in categorical_cols:
    tmp = (
        df.groupby([col, TARGET_COL])
          .size()
          .reset_index(name="n")
    )

    pivot = tmp.pivot_table(index=col, columns=TARGET_COL, values="n", fill_value=0)
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

    categorical_profile_all.append(pivot.head(TOP_N_CATEGORY_LEVELS))

if categorical_profile_all:
    categorical_profile_df = pd.concat(categorical_profile_all, ignore_index=True)
    print("\nTop livelli categorici associati a tappeto=1:")
    print(categorical_profile_df.head(40))
else:
    categorical_profile_df = pd.DataFrame()
    print("\nNessuna variabile categorica utile trovata.")

# ============================================================
# FEATURE IMPORTANCE DESCRITTIVA (MUTUAL INFORMATION)
# ============================================================

mi_results = []

X_mi = X.copy()

for col in X_mi.columns:
    if X_mi[col].dtype == "object":
        X_mi[col] = X_mi[col].astype(str).fillna("MISSING")
        X_mi[col] = X_mi[col].astype("category").cat.codes
    else:
        X_mi[col] = X_mi[col].fillna(X_mi[col].median())

try:
    mi_scores = mutual_info_classif(X_mi, y, discrete_features='auto', random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({
        "variabile": X_mi.columns,
        "mutual_info_score": mi_scores
    }).sort_values("mutual_info_score", ascending=False)

    print("\nTop variabili per Mutual Information:")
    print(mi_df.head(20))
except Exception as e:
    print("\nMutual Information non calcolabile:", e)
    mi_df = pd.DataFrame(columns=["variabile", "mutual_info_score"])

# ============================================================
# MODELLO PREDITTIVO
# ============================================================

# Se c'è una sola classe, niente training: si salva solo analisi descrittiva
if y.nunique() < 2:
    print("\n" + "=" * 80)
    print("ATTENZIONE: il target ha una sola classe.")
    print("Non è possibile allenare LogisticRegression.")
    print("Viene prodotta solo l'analisi descrittiva del profilo.")
    print("=" * 80)

    # salvataggi base
    numeric_profile_df.to_excel(os.path.join(OUTPUT_DIR, "profilo_numerico_tappeto.xlsx"), index=False)
    categorical_profile_df.to_excel(os.path.join(OUTPUT_DIR, "profilo_categorico_tappeto.xlsx"), index=False)
    mi_df.to_excel(os.path.join(OUTPUT_DIR, "mutual_info_tappeto.xlsx"), index=False)

    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "report_cluster_tappeto.xlsx"), engine="openpyxl") as writer:
        numeric_profile_df.to_excel(writer, sheet_name="profilo_numerico", index=False)
        categorical_profile_df.to_excel(writer, sheet_name="profilo_categorico", index=False)
        mi_df.to_excel(writer, sheet_name="mutual_info", index=False)

    print(f"\nReport salvato in: {OUTPUT_DIR}")

else:
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None
    )

    # preprocessing
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

    # modello
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    clf.fit(X_train, y_train)

    # predizioni
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 80)
    print("VALUTAZIONE MODELLO")
    print("=" * 80)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.4f}")
    except Exception as e:
        print("ROC AUC non calcolabile:", e)
        auc = np.nan

    # ========================================================
    # FEATURE IMPORTANCE DEL MODELLO
    # ========================================================

    # recupera nomi feature post one-hot
    try:
        feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
        coefficients = clf.named_steps["model"].coef_[0]

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients)
        }).sort_values("abs_coefficient", ascending=False)

        print("\nTop feature del modello:")
        print(coef_df.head(30)[["feature", "coefficient"]])

    except Exception as e:
        print("\nImpossibile estrarre i coefficienti del modello:", e)
        coef_df = pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    # ========================================================
    # SCORING SU TUTTO IL DATASET
    # ========================================================

    df_scored = df.copy()
    df_scored["prob_tappeto_1"] = clf.predict_proba(X)[:, 1]
    df_scored["pred_tappeto"] = clf.predict(X)

    top_clients = df_scored.sort_values("prob_tappeto_1", ascending=False).copy()

    # ========================================================
    # SALVATAGGI
    # ========================================================

    numeric_profile_df.to_excel(os.path.join(OUTPUT_DIR, "profilo_numerico_tappeto.xlsx"), index=False)
    categorical_profile_df.to_excel(os.path.join(OUTPUT_DIR, "profilo_categorico_tappeto.xlsx"), index=False)
    mi_df.to_excel(os.path.join(OUTPUT_DIR, "mutual_info_tappeto.xlsx"), index=False)
    coef_df.to_excel(os.path.join(OUTPUT_DIR, "feature_importance_modello.xlsx"), index=False)
    df_scored.to_excel(os.path.join(OUTPUT_DIR, "dataset_scored_tappeto.xlsx"), index=False)
    top_clients.to_excel(os.path.join(OUTPUT_DIR, "top_clienti_prob_tappeto.xlsx"), index=False)

    # report unico excel
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "report_cluster_tappeto.xlsx"), engine="openpyxl") as writer:
        numeric_profile_df.to_excel(writer, sheet_name="profilo_numerico", index=False)
        categorical_profile_df.to_excel(writer, sheet_name="profilo_categorico", index=False)
        mi_df.to_excel(writer, sheet_name="mutual_info", index=False)
        coef_df.to_excel(writer, sheet_name="feature_model", index=False)
        df_scored.to_excel(writer, sheet_name="dataset_scored", index=False)
        top_clients.head(500).to_excel(writer, sheet_name="top_500_clienti", index=False)

    # salva modello
    model_bundle = {
        "pipeline": clf,
        "feature_columns": list(X.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": TARGET_COL,
        "id_col": ID_COL
    }

    model_path = os.path.join(OUTPUT_DIR, "modello_cluster_tappeto.joblib")
    joblib.dump(model_bundle, model_path)

    print("\n" + "=" * 80)
    print("FILE SALVATI")
    print("=" * 80)
    print(f"Cartella output: {OUTPUT_DIR}")
    print(f"Modello salvato: {model_path}")

# ============================================================
# FUNZIONE PER RIUSARE IL MODELLO SU UNA NUOVA LISTA NDG
# ============================================================

def applica_modello_a_nuovo_file(model_path, new_file_path, output_file_path, sheet_name=0):
    """
    Carica un modello già salvato e lo applica a un nuovo file.
    Produce score/probabilità per tappeto=1.
    """
    bundle = joblib.load(model_path)
    clf = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    ext = os.path.splitext(new_file_path)[1].lower()
    if ext in [".xlsx", ".xlsm", ".xls"]:
        new_df = pd.read_excel(new_file_path, sheet_name=sheet_name)
    elif ext == ".csv":
        new_df = pd.read_csv(new_file_path)
    else:
        raise ValueError(f"Formato file non supportato: {ext}")

    # allinea colonne
    for col in feature_columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    X_new = new_df[feature_columns].copy()

    # conversione soft dei possibili numerici
    for col in X_new.columns:
        if X_new[col].dtype == "object":
            tmp = pd.to_numeric(X_new[col].astype(str).str.replace(",", "."), errors="coerce")
            if tmp.notna().mean() >= 0.70:
                X_new[col] = tmp

    new_df["prob_tappeto_1"] = clf.predict_proba(X_new)[:, 1]
    new_df["pred_tappeto"] = clf.predict(X_new)

    new_df = new_df.sort_values("prob_tappeto_1", ascending=False)
    new_df.to_excel(output_file_path, index=False)

    print(f"\nNuovo scoring completato. File salvato in: {output_file_path}")
    return new_df

# ============================================================
# ESEMPIO DI RIUTILIZZO MODELLO
# ============================================================
# Decommenta e personalizza:
#
# applica_modello_a_nuovo_file(
#     model_path=os.path.join(OUTPUT_DIR, "modello_cluster_tappeto.joblib"),
#     new_file_path=r"C:\percorso\nuova_lista_ndg.xlsx",
#     output_file_path=os.path.join(OUTPUT_DIR, "nuova_lista_ndg_scored.xlsx"),
#     sheet_name=0
# )
