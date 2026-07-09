# ============================================================
# MODELLO CLUSTER GESTIONE CLIENTI IN SCONFINO
# Cluster30 / Cluster90 / Cluster180
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

from scipy.stats import kruskal, chi2_contingency


# ============================================================
# 1. PARAMETRI
# ============================================================

INPUT_FILE = "dataset_clienti_cluster.xlsx"
OUTPUT_FILE = "output_modello_cluster_gestione.xlsx"

ID_COL = "NDG2"
TARGET_COL = "cluster"

CLUSTER_ORDER = {
    "cluster30": 1,
    "cluster90": 2,
    "cluster90+": 3,
    "cluster180": 3
}


# ============================================================
# 2. CARICAMENTO DATASET
# ============================================================

df = pd.read_excel(INPUT_FILE)

print("Righe dataset:", len(df))
print("Colonne:", len(df.columns))


# ============================================================
# 3. NORMALIZZAZIONE TARGET
# ============================================================

df[TARGET_COL] = df[TARGET_COL].astype(str).str.lower().str.strip()

# Uniformo eventuali nomi
df[TARGET_COL] = df[TARGET_COL].replace({
    "cluster 30": "cluster30",
    "cluster 90": "cluster90",
    "cluster 90+": "cluster90+",
    "cluster180+": "cluster180"
})

df = df[df[TARGET_COL].isin(CLUSTER_ORDER.keys())].copy()

df["target_ordinale"] = df[TARGET_COL].map(CLUSTER_ORDER)

print(df[TARGET_COL].value_counts())


# ============================================================
# 4. SEPARAZIONE FEATURE / TARGET
# ============================================================

drop_cols = [TARGET_COL, "target_ordinale"]

if ID_COL in df.columns:
    drop_cols.append(ID_COL)

X = df.drop(columns=drop_cols, errors="ignore")
y = df[TARGET_COL]


# Rimuovo colonne completamente vuote
X = X.dropna(axis=1, how="all")


# ============================================================
# 5. IDENTIFICAZIONE VARIABILI NUMERICHE E CATEGORICHE
# ============================================================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("Variabili numeriche:", len(numeric_features))
print("Variabili categoriche:", len(categorical_features))


# ============================================================
# 6. PROFILING STATISTICO PER CLUSTER
# ============================================================

profile_numeric = df.groupby(TARGET_COL)[numeric_features].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

profile_categorical = {}

for col in categorical_features:
    tmp = pd.crosstab(df[TARGET_COL], df[col], normalize="index")
    profile_categorical[col] = tmp


# ============================================================
# 7. TEST STATISTICI
# ============================================================

stat_tests = []

# Numeriche: Kruskal-Wallis
for col in numeric_features:
    groups = [
        df.loc[df[TARGET_COL] == c, col].dropna()
        for c in df[TARGET_COL].unique()
    ]

    groups = [g for g in groups if len(g) > 5]

    if len(groups) >= 2:
        try:
            stat, p_value = kruskal(*groups)
            stat_tests.append({
                "variable": col,
                "type": "numeric",
                "test": "kruskal",
                "p_value": p_value
            })
        except Exception:
            pass


# Categoriche: Chi-square
for col in categorical_features:
    try:
        table = pd.crosstab(df[TARGET_COL], df[col])
        if table.shape[0] > 1 and table.shape[1] > 1:
            chi2, p_value, dof, expected = chi2_contingency(table)
            stat_tests.append({
                "variable": col,
                "type": "categorical",
                "test": "chi_square",
                "p_value": p_value
            })
    except Exception:
        pass

stat_tests_df = pd.DataFrame(stat_tests).sort_values("p_value")


# ============================================================
# 8. PREPROCESSING MODELLO
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# 10. MODELLO 1 - DECISION TREE INTERPRETABILE
# ============================================================

tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42
    ))
])

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

print("\n=== DECISION TREE ===")
print(classification_report(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))


# ============================================================
# 11. MODELLO 2 - RANDOM FOREST
# ============================================================

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)

print("\n=== RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))


# ============================================================
# 12. MODELLO 3 - GRADIENT BOOSTING
# ============================================================

gb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
y_proba_gb = gb_model.predict_proba(X_test)

print("\n=== GRADIENT BOOSTING ===")
print(classification_report(y_test, y_pred_gb))
print(confusion_matrix(y_test, y_pred_gb))


# ============================================================
# 13. FEATURE IMPORTANCE RANDOM FOREST
# ============================================================

def get_feature_names(preprocessor):
    feature_names = []

    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_names)

    return feature_names


rf_preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

feature_names = get_feature_names(rf_preprocessor)

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_classifier.feature_importances_
}).sort_values("importance", ascending=False)


# ============================================================
# 14. MANAGEMENT PERSISTENCE SCORE 0-100
# ============================================================

classes = rf_classifier.classes_

score_weights = {
    "cluster30": 30,
    "cluster90": 60,
    "cluster90+": 90,
    "cluster180": 100
}

proba_full = rf_model.predict_proba(X)

score_df = pd.DataFrame(proba_full, columns=[f"prob_{c}" for c in classes])

score = np.zeros(len(score_df))

for c in classes:
    score += score_df[f"prob_{c}"] * score_weights.get(c, 0)

score_df["management_persistence_score"] = score
score_df["predicted_cluster"] = rf_model.predict(X)

if ID_COL in df.columns:
    score_df[ID_COL] = df[ID_COL].values

score_df[TARGET_COL] = df[TARGET_COL].values


# ============================================================
# 15. REGOLE DECISION TREE
# ============================================================

tree_classifier = tree_model.named_steps["model"]
tree_preprocessor = tree_model.named_steps["preprocessor"]

tree_feature_names = get_feature_names(tree_preprocessor)

tree_rules = export_text(
    tree_classifier,
    feature_names=list(tree_feature_names)
)

print("\n=== REGOLE DECISION TREE ===")
print(tree_rules)


# ============================================================
# 16. PROFILO MEDIO PER CLUSTER
# ============================================================

cluster_summary = df.groupby(TARGET_COL)[numeric_features].median().T

cluster_summary["delta_180_vs_30"] = (
    cluster_summary.get("cluster180", cluster_summary.iloc[:, -1])
    - cluster_summary.get("cluster30", cluster_summary.iloc[:, 0])
)


# ============================================================
# 17. EXPORT EXCEL
# ============================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    profile_numeric.to_excel(writer, sheet_name="profiling_numeric")
    stat_tests_df.to_excel(writer, sheet_name="stat_tests", index=False)
    feature_importance.to_excel(writer, sheet_name="feature_importance", index=False)
    score_df.to_excel(writer, sheet_name="scores_clienti", index=False)
    cluster_summary.to_excel(writer, sheet_name="cluster_summary")

    pd.DataFrame({
        "decision_tree_rules": tree_rules.split("\n")
    }).to_excel(writer, sheet_name="decision_tree_rules", index=False)

print(f"\nFile esportato: {OUTPUT_FILE}")
