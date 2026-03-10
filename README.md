# ============================================================
# SCORING + PROFILAZIONE CLIENTI TAPPETO
# Versione Jupyter-friendly
# ============================================================

import re
import numpy as np
import pandas as pd

from IPython.display import display, Markdown

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # ===== input =====
    "path_xlsx": r"C:\path\tuo_file.xlsx",   # <--- CAMBIA QUI
    "sheet_name": 0,                         # nome sheet o indice

    # ===== colonne attese =====
    "col_ndg": "ndg",
    "col_month": "MESE",
    "col_cluster": "Cluster",

    # ===== output =====
    "output_scored": "ndg_scored_tappeto.xlsx",
    "output_profile": "profilazione_clienti_tappeto.xlsx",

    # ===== profiling opzionale su variabili originali =====
    # metti qui eventuali colonne categoriche del dataset sorgente
    "profile_categorical_cols": [
        "Cluster",
        # "Segmento",
        # "Area",
        # "Filiale",
        # "Prodotto",
        # "Canale",
    ],

    # metti qui eventuali colonne numeriche del dataset sorgente
    "profile_numeric_cols": [
        # "eta",
        # "saldo",
        # "giacenza_media",
        # "n_prodotti",
    ],
}

# ============================================================
# FUNZIONI UTILI JUPYTER
# ============================================================

def h1(txt):
    display(Markdown(f"# {txt}"))

def h2(txt):
    display(Markdown(f"## {txt}"))

def h3(txt):
    display(Markdown(f"### {txt}"))

def show(df, n=10):
    display(df.head(n))

# ============================================================
# NORMALIZZAZIONE NOMI COLONNA
# ============================================================

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def resolve_column(df, wanted_name):
    """
    Cerca una colonna in modo robusto:
    - match esatto
    - match case-insensitive
    - match trimmed/lower
    """
    cols = list(df.columns)

    if wanted_name in cols:
        return wanted_name

    lower_map = {str(c).strip().lower(): c for c in cols}
    key = str(wanted_name).strip().lower()

    if key in lower_map:
        return lower_map[key]

    raise KeyError(f"Colonna non trovata: {wanted_name}. Colonne disponibili: {cols}")

# ============================================================
# PARSE MESE
# Supporta:
# - ott-25
# - ott/25
# - ott 25
# - Oct-25
# - 2025-10
# - 10/2025
# - datetime già valido
# ============================================================

MONTH_MAP_IT = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12
}

MONTH_MAP_EN = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

def parse_month_to_period(x):
    if pd.isna(x):
        return pd.NaT

    # caso datetime / timestamp
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        try:
            ts = pd.to_datetime(x, errors="coerce")
            if pd.isna(ts):
                return pd.NaT
            return pd.Period(ts, freq="M")
        except Exception:
            return pd.NaT

    s = str(x).strip()

    if s == "":
        return pd.NaT

    s_low = s.lower()

    # 1) formato it/en tipo ott-25 / oct-25 / ott 25 / ott/25
    m = re.match(r"^([a-z]{3})[-/ ](\d{2})$", s_low)
    if m:
        mon_txt = m.group(1)
        yy = int(m.group(2))
        month = MONTH_MAP_IT.get(mon_txt, MONTH_MAP_EN.get(mon_txt))
        if month is not None:
            year = 2000 + yy
            return pd.Period(f"{year}-{month:02d}", freq="M")

    # 2) formato yyyy-mm
    m = re.match(r"^(\d{4})-(\d{1,2})$", s_low)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12:
            return pd.Period(f"{year}-{month:02d}", freq="M")

    # 3) formato mm/yyyy o m/yyyy
    m = re.match(r"^(\d{1,2})/(\d{4})$", s_low)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if 1 <= month <= 12:
            return pd.Period(f"{year}-{month:02d}", freq="M")

    # 4) fallback con pandas
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return pd.NaT
        return pd.Period(dt, freq="M")
    except Exception:
        return pd.NaT

# ============================================================
# TARGET TAPPETO
# ============================================================

def is_tappeto(val):
    if pd.isna(val):
        return False

    s = str(val).strip().lower()

    # robusto: intercetta qualsiasi cluster che contenga "tappeto"
    return "tappeto" in s

# ============================================================
# CLUSTER CHANGES
# numero cambi di cluster lungo i mesi (transizioni), non solo nunique-1
# ============================================================

def compute_cluster_changes(group, col_period, col_cluster):
    g = group[[col_period, col_cluster]].copy()
    g = g.dropna(subset=[col_period])

    if g.empty:
        return 0

    # se nello stesso mese ci sono più righe, tengo una sequenza mensile pulita
    g = g.sort_values(col_period)
    g[col_cluster] = g[col_cluster].astype(str).str.strip()

    # una riga per mese: prendo il primo cluster osservato nel mese
    g = g.groupby(col_period, as_index=False)[col_cluster].first()

    if len(g) <= 1:
        return 0

    changes = (g[col_cluster] != g[col_cluster].shift(1)).sum() - 1
    return int(max(changes, 0))

# ============================================================
# PROFILING
# ============================================================

def profile_categorical(df_in, col, target="tappeto_any"):
    t = (
        df_in.groupby(col, dropna=False)[target]
        .agg(n="count", rate="mean")
        .reset_index()
    )
    t["rate"] = (t["rate"] * 100).round(2)
    t = t.sort_values(["rate", "n"], ascending=[False, False])
    return t

def safe_sheet_name(name, max_len=31):
    name = re.sub(r"[:\\/?*\[\]]", "_", str(name))
    return name[:max_len]

# ============================================================
# MAIN
# ============================================================

def main():
    h1("SCORING + PROFILAZIONE CLIENTI TAPPETO")

    # ========================================================
    # 1) LETTURA
    # ========================================================
    h2("1. Lettura file")

    df = pd.read_excel(CONFIG["path_xlsx"], sheet_name=CONFIG["sheet_name"])
    df = normalize_columns(df)

    print("Shape iniziale:", df.shape)
    print("Colonne disponibili:")
    print(df.columns.tolist())

    # risoluzione robusta nomi colonna
    col_ndg = resolve_column(df, CONFIG["col_ndg"])
    col_month = resolve_column(df, CONFIG["col_month"])
    col_cluster = resolve_column(df, CONFIG["col_cluster"])

    print("\nColonne usate:")
    print("NDG    ->", col_ndg)
    print("MESE   ->", col_month)
    print("CLUSTER->", col_cluster)

    # ========================================================
    # 2) PREP
    # ========================================================
    h2("2. Preparazione dati")

    work = df.copy()

    work[col_ndg] = work[col_ndg].astype(str).str.strip()
    work[col_cluster] = work[col_cluster].astype(str).str.strip()

    work["MESE_PARSED"] = work[col_month].apply(parse_month_to_period)
    work["is_tappeto"] = work[col_cluster].apply(is_tappeto)

    print("Righe totali:", len(work))
    print("NDG non null:", work[col_ndg].notna().sum())
    print("MESE_PARSED validi:", work["MESE_PARSED"].notna().sum())
    print("Righe tappeto:", int(work["is_tappeto"].sum()))

    h3("Distribuzione cluster")
    display(work[col_cluster].value_counts(dropna=False).rename_axis("Cluster").reset_index(name="n").head(20))

    h3("Righe tappeto")
    display(work["is_tappeto"].value_counts(dropna=False).rename_axis("is_tappeto").reset_index(name="n"))

    # ========================================================
    # 3) AGGREGAZIONE COERENTE A LIVELLO NDG
    # ========================================================
    h2("3. Aggregazione NDG-level")

    def build_ndg_features(g):
        mesi_validi = g["MESE_PARSED"].dropna().unique()
        n_months = len(mesi_validi)

        mesi_tappeto = g.loc[g["is_tappeto"], "MESE_PARSED"].dropna().unique()
        n_months_tappeto = len(mesi_tappeto)

        tappeto_any = int(n_months_tappeto >= 1)
        tappeto_recur_ge2 = int(n_months_tappeto >= 2)

        cluster_changes = compute_cluster_changes(
            group=g,
            col_period="MESE_PARSED",
            col_cluster=col_cluster
        )

        return pd.Series({
            "n_months": n_months,
            "n_months_tappeto": n_months_tappeto,
            "tappeto_any": tappeto_any,
            "tappeto_recur_ge2": tappeto_recur_ge2,
            "cluster_changes": cluster_changes
        })

    scored = work.groupby(col_ndg, dropna=False).apply(build_ndg_features).reset_index()
    scored = scored.rename(columns={col_ndg: "ndg"})

    # score semplice, leggibile e interpretabile
    # prevale la ricorrenza tappeto, poi numero mesi tappeto, poi stabilità/cambi cluster
    scored["score"] = (
        0.55 * scored["tappeto_any"] +
        0.25 * scored["tappeto_recur_ge2"] +
        0.15 * np.where(scored["n_months"] > 0,
                        scored["n_months_tappeto"] / scored["n_months"],
                        0) +
        0.05 * np.clip(scored["cluster_changes"], 0, 5) / 5
    )

    scored["score"] = scored["score"].round(6)

    # ordinamento
    scored = scored.sort_values(
        ["score", "n_months_tappeto", "tappeto_recur_ge2", "cluster_changes"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    print("Shape scored:", scored.shape)

    h3("Controllo coerenza")
    incoerenti = scored[scored["n_months_tappeto"] > scored["n_months"]]
    print("Righe incoerenti (n_months_tappeto > n_months):", len(incoerenti))

    h3("Top clienti")
    show(scored, 20)

    # ========================================================
    # 4) PROFILO SU DATASET AGGREGATO
    # ========================================================
    h2("4. Profilazione base sul dataset aggregato")

    num_cols_scored = [
        "n_months",
        "n_months_tappeto",
        "tappeto_recur_ge2",
        "cluster_changes",
        "score"
    ]

    profile_num_scored = (
        scored.groupby("tappeto_any")[num_cols_scored]
        .agg(["count", "mean", "median", "min", "max"])
        .round(3)
    )

    summary_scored = pd.DataFrame({
        "n_clienti": scored.groupby("tappeto_any").size(),
        "n_months_media": scored.groupby("tappeto_any")["n_months"].mean(),
        "n_months_tappeto_media": scored.groupby("tappeto_any")["n_months_tappeto"].mean(),
        "cluster_changes_media": scored.groupby("tappeto_any")["cluster_changes"].mean(),
        "score_medio": scored.groupby("tappeto_any")["score"].mean(),
        "score_mediano": scored.groupby("tappeto_any")["score"].median(),
    }).round(3)

    h3("Summary scored")
    display(summary_scored)

    h3("Profilo numerico scored")
    display(profile_num_scored)

    # ========================================================
    # 5) PROFILAZIONE SU DATASET ORIGINALE
    # ========================================================
    h2("5. Profilazione sul dataset originale")

    df_prof = work.merge(
        scored[["ndg", "tappeto_any", "tappeto_recur_ge2", "score"]],
        left_on=col_ndg,
        right_on="ndg",
        how="left"
    )

    print("Shape df_prof:", df_prof.shape)

    # categoriche esistenti davvero nel df
    cat_cols = []
    for c in CONFIG["profile_categorical_cols"]:
        if c in df_prof.columns:
            cat_cols.append(c)
        else:
            # provo match robusto
            try:
                real_c = resolve_column(df_prof, c)
                cat_cols.append(real_c)
            except:
                pass

    # numeriche esistenti davvero nel df
    num_cols_orig = []
    for c in CONFIG["profile_numeric_cols"]:
        if c in df_prof.columns:
            num_cols_orig.append(c)
        else:
            try:
                real_c = resolve_column(df_prof, c)
                num_cols_orig.append(real_c)
            except:
                pass

    cat_profiles = {}
    for col in cat_cols:
        try:
            cat_profiles[col] = profile_categorical(df_prof, col, target="tappeto_any")
        except Exception as e:
            print(f"Profilazione categorica fallita per {col}: {e}")

    profile_num_orig = None
    if len(num_cols_orig) > 0:
        try:
            profile_num_orig = (
                df_prof.groupby("tappeto_any")[num_cols_orig]
                .agg(["mean", "median", "std", "min", "max"])
                .round(3)
            )
        except Exception as e:
            print("Profilazione numerica originale fallita:", e)

    # tabella utile: distribuzione mesi tappeto
    dist_mesi_tappeto = (
        scored["n_months_tappeto"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("n_months_tappeto")
        .reset_index(name="n_clienti")
    )

    # tabella utile: top score
    top_score = scored.head(100).copy()

    # ========================================================
    # 6) SALVATAGGIO EXCEL
    # ========================================================
    h2("6. Salvataggio output")

    # file scoring
    scored.to_excel(CONFIG["output_scored"], index=False)
    print(f"File salvato: {CONFIG['output_scored']}")

    # file profiling multi-sheet
    with pd.ExcelWriter(CONFIG["output_profile"], engine="openpyxl") as writer:
        summary_scored.to_excel(writer, sheet_name="summary_scored")
        profile_num_scored.to_excel(writer, sheet_name="profile_num_scored")
        dist_mesi_tappeto.to_excel(writer, sheet_name="dist_mesi_tappeto", index=False)
        top_score.to_excel(writer, sheet_name="top_100_score", index=False)

        # campione dati aggregati
        scored.head(5000).to_excel(writer, sheet_name="sample_scored", index=False)

        # numeriche originali
        if profile_num_orig is not None:
            profile_num_orig.to_excel(writer, sheet_name="profile_num_original")

        # categoriche originali
        for col, tab in cat_profiles.items():
            tab.to_excel(writer, sheet_name=safe_sheet_name(f"cat_{col}"), index=False)

        # mini export del dataset prof con target, utile per analisi successive
        export_cols = ["ndg", "tappeto_any", "tappeto_recur_ge2", "score", "MESE_PARSED", "is_tappeto"]
        export_cols = [c for c in export_cols if c in df_prof.columns]
        extra_cols = [col_ndg, col_month, col_cluster]
        extra_cols = [c for c in extra_cols if c in df_prof.columns and c not in export_cols]

        df_prof_export = df_prof[extra_cols + export_cols].head(50000).copy()
        df_prof_export.to_excel(writer, sheet_name="sample_original_with_target", index=False)

    print(f"File salvato: {CONFIG['output_profile']}")

    # ========================================================
    # 7) OUTPUT NOTEBOOK
    # ========================================================
    h2("7. Output finali notebook")

    h3("Distribuzione target tappeto_any")
    display(scored["tappeto_any"].value_counts(dropna=False).rename_axis("tappeto_any").reset_index(name="n_clienti"))

    h3("Distribuzione n_months_tappeto")
    display(dist_mesi_tappeto)

    if len(cat_profiles) > 0:
        first_key = list(cat_profiles.keys())[0]
        h3(f"Esempio profilo categorico: {first_key}")
        display(cat_profiles[first_key].head(20))

    if profile_num_orig is not None:
        h3("Profilo numerico dataset originale")
        display(profile_num_orig)

    h2("Completato")
    return scored, df_prof

# ============================================================
# RUN
# ============================================================

scored, df_prof = main()
