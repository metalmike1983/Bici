# =========================================================
# SCORING CLIENTI CLUSTER TAPPETO - VERSIONE JUPYTER
# =========================================================

import re
import warnings
import numpy as np
import pandas as pd

from IPython.display import display, Markdown

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# =========================================================
# CONFIG
# =========================================================

CONFIG = {

    "path_xlsx": r"C:\Users\D75737\OneDrive - BNP Paribas\Bureau\GP_oct_nov_dic_2025.xlsx",
    "sheet_name": "GP_OCT_NOV_DIC_25",

    "col_ndg": "ndg",
    "col_month": "MESE",
    "col_cluster": "Cluster",

    "tappeto_regex": r"\btappeto\b",

    "min_months_for_true_tappeto": 1,

    "score_threshold": 0.65
}

MONTH_MAP_IT = {
    "gen":1,"feb":2,"mar":3,"apr":4,"mag":5,"giu":6,
    "lug":7,"ago":8,"set":9,"ott":10,"nov":11,"dic":12
}


# =========================================================
# FUNZIONI UTILI JUPYTER
# =========================================================

def h1(txt):
    display(Markdown(f"# {txt}"))

def h2(txt):
    display(Markdown(f"## {txt}"))

def show(df,n=10):
    display(df.head(n))


# =========================================================
# PARSE MESE
# =========================================================

def parse_month_to_period(x):

    if pd.isna(x):
        return pd.NaT

    s=str(x).strip().lower()

    m=re.match(r"^([a-z]{3})[-/ ](\d{2})$",s)

    if m and m.group(1) in MONTH_MAP_IT:

        month=MONTH_MAP_IT[m.group(1)]
        year=2000+int(m.group(2))

        return pd.Period(f"{year}-{month:02d}",freq="M")

    return pd.NaT


# =========================================================
# CLUSTER TAPPETO
# =========================================================

def is_tappeto(val):

    if pd.isna(val):
        return False

    return re.search(CONFIG["tappeto_regex"],str(val).lower()) is not None


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def build_features(df):

    col_ndg=CONFIG["col_ndg"]
    col_month=CONFIG["col_month"]
    col_cluster=CONFIG["col_cluster"]

    df=df.copy()

    df[col_ndg]=df[col_ndg].astype(str).str.strip()

    df["_month"]=df[col_month].apply(parse_month_to_period)

    df["_is_tappeto"]=df[col_cluster].apply(is_tappeto)

    h2("Distribuzione cluster")

    show(df[col_cluster].value_counts().reset_index())

    h2("Righe tappeto")

    show(df["_is_tappeto"].value_counts().reset_index())

    g=df.groupby(col_ndg)

    months=g["_month"].nunique().rename("n_months")

    tappeto_months=g["_is_tappeto"].sum().rename("n_months_tappeto")

    tappeto_any=(tappeto_months>=1).astype(int).rename("tappeto_any")

    tappeto_rec=(tappeto_months>=2).astype(int).rename("tappeto_recur_ge2")

    cluster_changes=g[col_cluster].apply(
        lambda s:(s!=s.shift()).sum()-1
    ).rename("cluster_changes")

    feat=pd.concat([
        months,
        tappeto_months,
        tappeto_any,
        tappeto_rec,
        cluster_changes
    ],axis=1).reset_index()

    return feat


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(feat,target):

    y=feat[target]

    if y.nunique()<2:

        print("Target con una sola classe → fallback scoring")

        scored=feat.copy()

        scored["score"]=scored["n_months_tappeto"]/scored["n_months_tappeto"].max()

        return None,scored

    X=feat.drop(columns=[target])

    num_cols=X.select_dtypes(include=np.number).columns.tolist()

    cat_cols=[c for c in X.columns if c not in num_cols]

    pre=ColumnTransformer(

        transformers=[

            ("num",
             Pipeline([
                 ("imp",SimpleImputer(strategy="median")),
                 ("scaler",StandardScaler())
             ]),
             num_cols
            ),

            ("cat",
             Pipeline([
                 ("imp",SimpleImputer(strategy="most_frequent")),
                 ("onehot",OneHotEncoder(handle_unknown="ignore"))
             ]),
             cat_cols
            )
        ]
    )

    pipe=Pipeline([

        ("pre",pre),

        ("model",LogisticRegression(max_iter=2000))
    ])

    X_train,X_test,y_train,y_test=train_test_split(

        X,y,test_size=0.25,random_state=42,stratify=y

    )

    pipe.fit(X_train,y_train)

    proba=pipe.predict_proba(X_test)[:,1]

    auc=roc_auc_score(y_test,proba)

    print("AUC:",round(auc,3))

    all_proba=pipe.predict_proba(X)[:,1]

    scored=feat.copy()

    scored["score"]=all_proba

    return pipe,scored


# =========================================================
# MAIN
# =========================================================

def main():

    h1("SCORING CLIENTI TAPPETO")

    df=pd.read_excel(

        CONFIG["path_xlsx"],
        sheet_name=CONFIG["sheet_name"]
    )

    print("Shape:",df.shape)

    feat=build_features(df)

    target="tappeto_recur_ge2" if CONFIG["min_months_for_true_tappeto"]>=2 else "tappeto_any"

    print("Target:",target)

    model,scored=train_model(feat,target)

    scored=scored.sort_values(

        ["score","n_months_tappeto"],

        ascending=False

    )

    h2("Top clienti")

    show(scored,30)

    scored.to_excel("ndg_scored_tappeto.xlsx",index=False)

    print("File salvato: ndg_scored_tappeto.xlsx")

    return scored


# =========================================================
# RUN
# =========================================================

scored = main()
