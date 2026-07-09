# ============================================================
# MODELLO PROFILAZIONE CLIENTI IN SCONFINO
# Cluster30 / Cluster90 / Cluster90+
# Versione V2 con:
# - rimozione leakage
# - profiling cluster
# - test statistici
# - decision tree
# - random forest
# - gradient boosting
# - feature importance
# - score 0-100
# - cliente tipo
# - business insight automatici
# - export Excel finale
# ============================================================

import pandas as pd
import numpy as np

from scipy.stats import kruskal, chi2_contingency

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer


# ============================================================
# 1. PARAMETRI
# ============================================================

INPUT_FILE = "dataset_clienti_cluster.xlsx"
OUTPUT_FILE = "output_modello_cluster_gestione_v2.xlsx"

TARGET_COL = "cluster"

LEAKAGE_COLS = [
    "gestione",
    "giorni_gestione",
    "max_giorni_gestione",
    "num_gestioni",
    "giorni_totali_gestione"
]

ID_COLS = [
    "NDG",
    "NDG2",
    "ndg",
    "ndg2",
    "ID_CLIENTE",
    "id_cliente"
]

TARGET_DERIVED_COLS = [
    "target_ordinale"
]


# ============================================================
# 2. CARICAMENTO
# ============================================================

df = pd.read_excel(INPUT_FILE)

print("Righe dataset:", len(df))
print("Colonne iniziali:", len(df.columns))


# ============================================================
# 3. NORMALIZZAZIONE TARGET
# ============================================================

df[TARGET_COL] = df[TARGET_COL].astype(str).str.lower().str.strip()

df[TARGET_COL] = df[TARGET_COL].replace({
    "cluster 30": "cluster30",
    "cluster_30": "cluster30",
    "cluster 90": "cluster90",
    "cluster_90": "cluster90",
    "cluster 90+": "cluster90+",
    "cluster_90+": "cluster90+",
    "cluster180": "cluster90+",
    "cluster180+": "cluster90+"
})

valid_clusters = ["cluster30", "cluster90", "cluster90+"]
df = df[df[TARGET_COL].isin(valid_clusters)].copy()

cluster_rank = {
    "cluster30": 30,
    "cluster90": 60,
    "cluster90+": 100
}

df["target_ordinale"] = df[TARGET_COL].map(cluster_rank)

print("\nDistribuzione cluster:")
print(df[TARGET_COL].value_counts())


# ============================================================
# 4. RIMOZIONE VARIABILI DA NON USARE
# ============================================================

drop_cols = (
    [TARGET_COL]
    + TARGET_DERIVED_COLS
    + LEAKAGE_COLS
    + ID_COLS
)

X = df.drop(columns=drop_cols, errors="ignore")
y = df[TARGET_COL]

X = X.dropna(axis=1, how="all")

print("\nColonne dopo pulizia:", len(X.columns))

print("\nControllo colonne escluse ancora presenti:")
for c in LEAKAGE_COLS + ID_COLS:
    if c in X.columns:
        print("ATTENZIONE ancora presente:", c)


# ============================================================
# 5. IDENTIFICAZIONE NUMERICHE / CATEGORICHE
# ============================================================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nVariabili numeriche:", len(numeric_features))
print("Variabili categoriche:", len(categorical_features))


# ============================================================
# 6. PROFILING NUMERICO PER CLUSTER
# ============================================================

profile_numeric = df.groupby(TARGET_COL)[numeric_features].agg([
    "count", "mean", "median", "std", "min", "max"
])

cluster_median = df.groupby(TARGET_COL)[numeric_features].median().T
cluster_mean = df.groupby(TARGET_COL)[numeric_features].mean().T

for c in valid_clusters:
    if c not in cluster_median.columns:
        cluster_median[c] = np.nan

cluster_profile = cluster_median[valid_clusters].copy()

if "cluster30" in cluster_profile.columns and "cluster90+" in cluster_profile.columns:
    cluster_profile["delta_90plus_vs_30"] = (
        cluster_profile["cluster90+"] - cluster_profile["cluster30"]
    )

    cluster_profile["abs_delta_90plus_vs_30"] = (
        cluster_profile["delta_90plus_vs_30"].abs()
    )


# ============================================================
# 7. TEST STATISTICI
# ============================================================

stat_tests = []

for col in numeric_features:
    groups = [
        df.loc[df[TARGET_COL] == c, col].dropna()
        for c in valid_clusters
        if c in df[TARGET_COL].unique()
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

stat_tests_df = pd.DataFrame(stat_tests)

if len(stat_tests_df) > 0:
    stat_tests_df = stat_tests_df.sort_values("p_value")
    stat_tests_df["significant_005"] = stat_tests_df["p_value"] < 0.05
    stat_tests_df["significance"] = pd.cut(
        stat_tests_df["p_value"],
        bins=[-1, 0.001, 0.01, 0.05, 1],
        labels=["***", "**", "*", ""]
    )


# ============================================================
# 8. PREPROCESSING
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
# 10. MODELLI
# ============================================================

tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=80,
        class_weight="balanced",
        random_state=42
    ))
])

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=40,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

gb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        random_state=42
    ))
])

models = {
    "Decision Tree": tree_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

model_reports = {}
confusion_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"\n=== {name.upper()} ===")
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))

    model_reports[name] = classification_report(
        y_test,
        pred,
        output_dict=True
    )

    confusion_matrices[name] = pd.DataFrame(
        confusion_matrix(y_test, pred),
        index=[f"actual_{c}" for c in model.named_steps["model"].classes_],
        columns=[f"pred_{c}" for c in model.named_steps["model"].classes_]
    )


# ============================================================
# 11. CROSS VALIDATION RANDOM FOREST
# ============================================================

cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="f1_weighted"
)

cv_summary = pd.DataFrame({
    "metric": ["cv_f1_weighted_mean", "cv_f1_weighted_std"],
    "value": [cv_scores.mean(), cv_scores.std()]
})


# ============================================================
# 12. FEATURE NAMES COMPATIBILE CON SKLEARN VECCHIO
# ============================================================

def get_feature_names(preprocessor):
    feature_names = []

    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

        try:
            cat_names = cat_encoder.get_feature_names_out(categorical_features)
        except AttributeError:
            cat_names = cat_encoder.get_feature_names(categorical_features)

        feature_names.extend(cat_names)

    return feature_names


# ============================================================
# 13. FEATURE IMPORTANCE RANDOM FOREST
# ============================================================

rf_preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

feature_names = get_feature_names(rf_preprocessor)

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_classifier.feature_importances_
}).sort_values("importance", ascending=False)

top_features = feature_importance.head(30)


# ============================================================
# 14. SCORE 0-100
# ============================================================

rf_classes = rf_classifier.classes_
proba_full = rf_model.predict_proba(X)

score_df = pd.DataFrame(
    proba_full,
    columns=[f"prob_{c}" for c in rf_classes]
)

score = np.zeros(len(score_df))

for c in rf_classes:
    score += score_df[f"prob_{c}"] * cluster_rank.get(c, 0)

score_df["management_persistence_score"] = score
score_df["predicted_cluster"] = rf_model.predict(X)
score_df["actual_cluster"] = df[TARGET_COL].values

for id_col in ID_COLS:
    if id_col in df.columns:
        score_df[id_col] = df[id_col].values

score_df["score_band"] = pd.cut(
    score_df["management_persistence_score"],
    bins=[0, 40, 70, 100],
    labels=["Low persistence", "Medium persistence", "High persistence"],
    include_lowest=True
)


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
# 16. PROFILO CLIENTE TIPO PER CLUSTER
# ============================================================

cliente_tipo_rows = []

for cluster in valid_clusters:
    if cluster not in df[TARGET_COL].unique():
        continue

    subset = df[df[TARGET_COL] == cluster]

    for col in numeric_features:
        cliente_tipo_rows.append({
            "cluster": cluster,
            "variable": col,
            "mean": subset[col].mean(),
            "median": subset[col].median(),
            "p25": subset[col].quantile(0.25),
            "p75": subset[col].quantile(0.75),
            "missing_rate": subset[col].isna().mean()
        })

cliente_tipo_df = pd.DataFrame(cliente_tipo_rows)


# ============================================================
# 17. BUSINESS INSIGHT AUTOMATICI
# ============================================================

insights = []

if len(stat_tests_df) > 0:
    significant_vars = stat_tests_df[
        stat_tests_df["significant_005"] == True
    ]["variable"].tolist()
else:
    significant_vars = []

top_important_vars = top_features["feature"].head(20).tolist()

common_vars = [
    v for v in top_important_vars
    if v in numeric_features and v in significant_vars
]

for var in common_vars[:15]:
    try:
        c30 = cluster_profile.loc[var, "cluster30"]
        c90 = cluster_profile.loc[var, "cluster90"]
        c90p = cluster_profile.loc[var, "cluster90+"]

        if pd.notna(c30) and pd.notna(c90p):
            direction = "higher" if c90p > c30 else "lower"

            insights.append({
                "variable": var,
                "business_insight": (
                    f"{var}: median value is {direction} in cluster90+ "
                    f"than in cluster30. "
                    f"cluster30={c30:.2f}, cluster90={c90:.2f}, cluster90+={c90p:.2f}."
                )
            })
    except Exception:
        pass

business_insights_df = pd.DataFrame(insights)


# ============================================================
# 18. PROFILO COMPARATIVO CLUSTER
# ============================================================

cluster_comparison = cluster_profile.copy()

if len(stat_tests_df) > 0:
    pvals = stat_tests_df.set_index("variable")["p_value"].to_dict()
    cluster_comparison["p_value"] = cluster_comparison.index.map(pvals)
    cluster_comparison["significant_005"] = cluster_comparison["p_value"] < 0.05

cluster_comparison = cluster_comparison.sort_values(
    "abs_delta_90plus_vs_30",
    ascending=False
)


# ============================================================
# 19. REPORT MODELLI IN DATAFRAME
# ============================================================

model_report_rows = []

for model_name, report in model_reports.items():
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            row = {"model": model_name, "label": label}
            row.update(metrics)
            model_report_rows.append(row)
        else:
            model_report_rows.append({
                "model": model_name,
                "label": label,
                "value": metrics
            })

model_report_df = pd.DataFrame(model_report_rows)


# ============================================================
# 20. EXPORT EXCEL
# ============================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    pd.DataFrame(df[TARGET_COL].value_counts()).to_excel(
        writer,
        sheet_name="00_cluster_distribution"
    )

    model_report_df.to_excel(
        writer,
        sheet_name="01_model_performance",
        index=False
    )

    cv_summary.to_excel(
        writer,
        sheet_name="02_cross_validation",
        index=False
    )

    feature_importance.to_excel(
        writer,
        sheet_name="03_feature_importance",
        index=False
    )

    cluster_comparison.to_excel(
        writer,
        sheet_name="04_cluster_comparison"
    )

    stat_tests_df.to_excel(
        writer,
        sheet_name="05_stat_tests",
        index=False
    )

    cliente_tipo_df.to_excel(
        writer,
        sheet_name="06_cliente_tipo",
        index=False
    )

    score_df.to_excel(
        writer,
        sheet_name="07_scores_clienti",
        index=False
    )

    business_insights_df.to_excel(
        writer,
        sheet_name="08_business_insights",
        index=False
    )

    pd.DataFrame({
        "decision_tree_rules": tree_rules.split("\n")
    }).to_excel(
        writer,
        sheet_name="09_decision_tree_rules",
        index=False
    )

    for name, cm in confusion_matrices.items():
        safe_name = name.replace(" ", "_")[:20]
        cm.to_excel(
            writer,
            sheet_name=f"CM_{safe_name}"
        )

print(f"\nFile esportato: {OUTPUT_FILE}")
