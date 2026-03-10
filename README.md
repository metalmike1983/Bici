import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


# =========================
# CONFIG: adatta questi nomi
# =========================
CONFIG = {
    "path_xlsx": r"C:\Users\D75737\OneDrive - BNP Paribas\Bureau\GP_oct_nov_dic_2025.xlsx",
    "sheet_name": "GP_OCT_NOV_DIC_25",
    "col_ndg": "ndg",
    "col_month": "MESE",
    "col_cluster": "Cluster",

    # regex più robusto: trova qualsiasi cluster che contenga "tappeto"
    "tappeto_regex": r"\btappeto\b",

    # se 1: basta almeno 1 mese tappeto
    # se 2: almeno 2 mesi tappeto
    "min_months_for_true_tappeto": 1,

    "score_threshold": 0.65
}


MONTH_MAP_IT = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12
}


def parse_month_to_period(x):
    if pd.isna(x):
        return pd.NaT

    s = str(x).strip().lower()

    # es: ott-25 / ott 25 / ott/25
    m = re.match(r"^([a-z]{3})[-/ ](\d{2})$", s)
    if m and m.group(1) in MONTH_MAP_IT:
        month = MONTH_MAP_IT[m.group(1)]
        year = 2000 + int(m.group(2))
        return pd.Period(f"{year:04d}-{month:02d}", freq="M")

    # es: 2025-10 / 2025/10
    m = re.match(r"^(\d{4})[-/](\d{1,2})$", s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        return pd.Period(f"{year:04d}-{month:02d}", freq="M")

    # es: 10/2025
    m = re.match(r"^(\d{1,2})/(\d{4})$", s)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        return pd.Period(f"{year:04d}-{month:02d}", freq="M")

    return pd.NaT


def is_tappeto(cluster_value, tappeto_regex):
    if pd.isna(cluster_value):
        return False
    return re.search(tappeto_regex, str(cluster_value).strip().lower()) is not None


def build_ndg_features(df, col_ndg, col_month, col_cluster, tappeto_regex):
    df = df.copy()

    df[col_ndg] = df[col_ndg].astype(str).str.strip()
    df["_month"] = df[col_month].apply(parse_month_to_period)
    df["_cluster_clean"] = df[col_cluster].astype(str).str.strip().str.lower()
    df["_is_tappeto_row"] = df[col_cluster].apply(lambda x: is_tappeto(x, tappeto_regex))

    # diagnostica iniziale
    print("\n=== TOP 20 valori Cluster ===")
    print(df[col_cluster].astype(str).value_counts(dropna=False).head(20))

    print("\n=== TOP 20 valori Cluster puliti ===")
    print(df["_cluster_clean"].value_counts(dropna=False).head(20))

    print("\n=== Conteggio righe tappeto ===")
    print(df["_is_tappeto_row"].value_counts(dropna=False))

    n_bad_month = df["_month"].isna().sum()
    if n_bad_month > 0:
        print(f"\nATTENZIONE: {n_bad_month} righe con mese non parsato correttamente.")

    df = df.sort_values([col_ndg, "_month"])

    excluded = {col_ndg, col_month, col_cluster, "_month", "_is_tappeto_row", "_cluster_clean"}

    numeric_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    cat_cols = [
        c for c in df.columns
        if c not in excluded and not pd.api.types.is_numeric_dtype(df[c])
    ]

    g = df.groupby(col_ndg, dropna=False)

    months_count = g["_month"].nunique().rename("n_months_observed")
    tappeto_months = g.apply(
        lambda x: x.loc[x["_is_tappeto_row"], "_month"].nunique()
    ).rename("n_months_tappeto")

    tappeto_any = (tappeto_months >= 1).astype(int).rename("tappeto_any")
    tappeto_recidive = (tappeto_months >= 2).astype(int).rename("tappeto_recur_ge2")

    cluster_changes = g[col_cluster].apply(
        lambda s: int((s.astype(str).fillna("").ne(s.astype(str).fillna("").shift(1))).sum() - 1) if len(s) > 1 else 0
    ).rename("n_cluster_changes")

    feats_num = []
    if numeric_cols:
        mean_ = g[numeric_cols].mean().add_prefix("mean_")
        std_ = g[numeric_cols].std(ddof=0).fillna(0).add_prefix("std_")
        last_ = g[numeric_cols].last().add_prefix("last_")
        first_ = g[numeric_cols].first().add_prefix("first_")

        delta_ = pd.DataFrame(
            last_.values - first_.values,
            index=last_.index,
            columns=[c.replace("last_", "delta_") for c in last_.columns]
        )

        feats_num = [mean_, std_, last_, delta_]

    feats_cat = []
    if cat_cols:
        def mode_safe(s):
            s = s.dropna().astype(str)
            if len(s) == 0:
                return np.nan
            return s.value_counts().index[0]

        mode_ = g[cat_cols].agg(mode_safe).add_prefix("mode_")
        feats_cat = [mode_]

    parts = [months_count, tappeto_months, tappeto_any, tappeto_recidive, cluster_changes]
    parts += feats_num
    parts += feats_cat

    feat_df = pd.concat(parts, axis=1).reset_index()

    return feat_df


def train_and_score(feat_df, target_col="tappeto_recur_ge2"):
    y = feat_df[target_col].astype(int)
    X = feat_df.drop(columns=[target_col])

    print("\n=== Distribuzione target ===")
    print(y.value_counts(dropna=False))

    # Se c'è una sola classe, niente training ML
    if y.nunique() < 2:
        print("\nATTENZIONE: il target contiene una sola classe. Salto il training del modello.")

        scored = feat_df[[CONFIG["col_ndg"]]].copy()

        # score diagnostico semplice: più mesi tappeto => score più alto
        if "n_months_tappeto" in feat_df.columns and feat_df["n_months_tappeto"].max() > 0:
            scored["score_tappeto"] = feat_df["n_months_tappeto"] / feat_df["n_months_tappeto"].max()
        else:
            scored["score_tappeto"] = 0.0

        scored["pred_tappeto"] = (scored["score_tappeto"] >= CONFIG["score_threshold"]).astype(int)

        diag_cols = [c for c in feat_df.columns if c in [
            "n_months_observed", "n_months_tappeto", "tappeto_any", "n_cluster_changes"
        ]]

        scored = scored.merge(
            feat_df[[CONFIG["col_ndg"]] + diag_cols],
            on=CONFIG["col_ndg"],
            how="left"
        )

        return None, scored

    num_cols = [
        c for c in X.columns
        if c != CONFIG["col_ndg"] and pd.api.types.is_numeric_dtype(X[c])
    ]
    cat_cols = [
        c for c in X.columns
        if c != CONFIG["col_ndg"] and not pd.api.types.is_numeric_dtype(X[c])
    ]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    # ulteriore check sul training set
    if y_train.nunique() < 2:
        print("\nATTENZIONE: dopo train/test split il training set contiene una sola classe.")
        print("Salto il training del modello e produco score diagnostico.")

        scored = feat_df[[CONFIG["col_ndg"]]].copy()

        if "n_months_tappeto" in feat_df.columns and feat_df["n_months_tappeto"].max() > 0:
            scored["score_tappeto"] = feat_df["n_months_tappeto"] / feat_df["n_months_tappeto"].max()
        else:
            scored["score_tappeto"] = 0.0

        scored["pred_tappeto"] = (scored["score_tappeto"] >= CONFIG["score_threshold"]).astype(int)

        diag_cols = [c for c in feat_df.columns if c in [
            "n_months_observed", "n_months_tappeto", "tappeto_any", "n_cluster_changes"
        ]]

        scored = scored.merge(
            feat_df[[CONFIG["col_ndg"]] + diag_cols],
            on=CONFIG["col_ndg"],
            how="left"
        )

        return None, scored

    pipe.fit(X_train, y_train)

    if y_test.nunique() > 1:
        proba_test = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba_test)
        print(f"\nAUC (target={target_col}): {auc:.3f}")
        print(classification_report(y_test, (proba_test >= 0.5).astype(int)))

    all_proba = pipe.predict_proba(X)[:, 1]

    scored = feat_df[[CONFIG["col_ndg"]]].copy()
    scored["score_tappeto"] = all_proba
    scored["pred_tappeto"] = (scored["score_tappeto"] >= CONFIG["score_threshold"]).astype(int)

    diag_cols = [c for c in feat_df.columns if c in [
        "n_months_observed", "n_months_tappeto", "tappeto_any", "n_cluster_changes"
    ]]

    scored = scored.merge(
        feat_df[[CONFIG["col_ndg"]] + diag_cols],
        on=CONFIG["col_ndg"],
        how="left"
    )

    return pipe, scored


def main():
    df = pd.read_excel(CONFIG["path_xlsx"], sheet_name=CONFIG["sheet_name"])

    print("Shape input:", df.shape)
    print("Colonne disponibili:")
    print(list(df.columns))

    feat_df = build_ndg_features(
        df=df,
        col_ndg=CONFIG["col_ndg"],
        col_month=CONFIG["col_month"],
        col_cluster=CONFIG["col_cluster"],
        tappeto_regex=CONFIG["tappeto_regex"]
    )

    target = "tappeto_recur_ge2" if CONFIG["min_months_for_true_tappeto"] >= 2 else "tappeto_any"

    print(f"\nTarget scelto: {target}")
    print(feat_df[target].value_counts(dropna=False))

    model, scored = train_and_score(feat_df, target_col=target)

    out_scored = "ndg_scored_tappeto.xlsx"
    scored.sort_values(["score_tappeto", "n_months_tappeto"], ascending=[False, False]).to_excel(out_scored, index=False)

    print(f"\nSalvato ranking: {out_scored}")


if __name__ == "__main__":
    main()
