def extract_model_coefficients(clf, numeric_cols=None, categorical_cols=None):
    """
    Estrae i coefficienti del modello finale dalla pipeline.
    Compatibile anche con versioni sklearn meno recenti.
    """
    try:
        preprocessor = clf.named_steps["preprocessor"]
        model = clf.named_steps["model"]

        coefficients = model.coef_[0]

        feature_names = []

        # scorri i transformer del ColumnTransformer
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder" and transformer == "drop":
                continue

            if isinstance(cols, slice):
                cols = list(range(len(cols)))

            # Pipeline numerica
            if name == "num":
                feature_names.extend(cols)

            # Pipeline categorica
            elif name == "cat":
                # se è una pipeline, prendo il onehot
                if hasattr(transformer, "named_steps"):
                    onehot = transformer.named_steps.get("onehot", None)
                else:
                    onehot = None

                if onehot is not None:
                    # sklearn vecchio
                    if hasattr(onehot, "get_feature_names"):
                        cat_names = onehot.get_feature_names(cols)
                    # sklearn nuovo
                    elif hasattr(onehot, "get_feature_names_out"):
                        cat_names = onehot.get_feature_names_out(cols)
                    else:
                        cat_names = cols
                    feature_names.extend(cat_names)
                else:
                    feature_names.extend(cols)

            else:
                feature_names.extend(cols)

        # sicurezza: se le lunghezze non coincidono, tronca alla minima
        if len(feature_names) != len(coefficients):
            min_len = min(len(feature_names), len(coefficients))
            print(
                f"Attenzione: feature_names={len(feature_names)}, coefficients={len(coefficients)}. "
                f"Allineo alla lunghezza minima={min_len}."
            )
            feature_names = list(feature_names)[:min_len]
            coefficients = coefficients[:min_len]

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients
        })

        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
        return coef_df

    except Exception as e:
        print(f"Impossibile estrarre i coefficienti del modello: {e}")
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

# =========================================================
# FUNZIONI DI SUPPORTO
# =========================================================

def load_data(file_path, sheet_name=0):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls", ".xlsm"]:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    elif ext == ".csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Formato file non supportato: {ext}")


def soft_convert_numeric(series, threshold=0.70):
    """
    Prova a convertire una colonna object in numerica.
    Converte solo se almeno 'threshold' dei valori è interpretabile come numero.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    s = series.astype(str).str.strip()

    # normalizzazione leggera
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(".", "", regex=False)   # es. 1.234 -> 1234
    s = s.str.replace(",", ".", regex=False)  # es. 12,5 -> 12.5

    converted = pd.to_numeric(s, errors="coerce")

    if converted.notna().mean() >= threshold:
        return converted

    return series


def build_feature_lists(X, min_unique_for_numeric=15):
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            nunique = X[col].nunique(dropna=True)
            if nunique <= min_unique_for_numeric:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def extract_model_coefficients(clf):
    """
    Estrae i coefficienti del modello finale dalla pipeline,
    usando i nomi feature reali post-preprocessing.
    """
    try:
        preprocessor = clf.named_steps["preprocessor"]
        model = clf.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()
        coefficients = model.coef_[0]

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients
        })

        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
        return coef_df

    except Exception as e:
        print(f"Impossibile estrarre i coefficienti del modello: {e}")
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])


def save_excel_report(path, sheets_dict):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets_dict.items():
            if df_sheet is None:
                continue
            if not isinstance(df_sheet, pd.DataFrame):
                df_sheet = pd.DataFrame(df_sheet)
            df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def prepara_X_nuovo_file(new_df, bundle):
    """
    Allinea il nuovo file alle feature usate in training.
    """
    feature_columns = bundle["feature_columns"]

    new_df = new_df.copy()
    new_df.columns = [str(c).strip() for c in new_df.columns]

    for col in feature_columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    X_new = new_df[feature_columns].copy()

    for col in X_new.columns:
        X_new[col] = soft_convert_numeric(X_new[col])

    return X_new, new_df


def applica_modello_a_nuovo_file(model_path, new_file_path, output_file_path, sheet_name=0):
    """
    Carica un modello già salvato e lo applica a un nuovo file.
    Produce score/probabilità per tappeto=1.
    """
    bundle = joblib.load(model_path)
    clf = bundle["pipeline"]

    ext = os.path.splitext(new_file_path)[1].lower()
    if ext in [".xlsx", ".xls", ".xlsm"]:
        new_df = pd.read_excel(new_file_path, sheet_name=sheet_name)
    elif ext == ".csv":
        new_df = pd.read_csv(new_file_path)
    else:
        raise ValueError(f"Formato file non supportato: {ext}")

    X_new, new_df = prepara_X_nuovo_file(new_df, bundle)

    if hasattr(clf, "predict_proba"):
        new_df["prob_tappeto_1"] = clf.predict_proba(X_new)[:, 1]
    else:
        new_df["prob_tappeto_1"] = np.nan

    new_df["pred_tappeto"] = clf.predict(X_new)

    new_df = new_df.sort_values("prob_tappeto_1", ascending=False).reset_index(drop=True)
    new_df.to_excel(output_file_path, index=False)

    print(f"\nNuovo scoring completato. File salvato in: {output_file_path}")
    display(new_df.head(20))

    return new_df


# =========================================================
# 1. LETTURA DATI
# =========================================================

df = load_data(INPUT_FILE, SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]

print("=" * 90)
print("DATASET CARICATO")
print("=" * 90)
print("Shape iniziale:", df.shape)
display(df.head())


# =========================================================
# 2. CONTROLLI PRELIMINARI
# =========================================================

if TARGET_COL not in df.columns:
    raise ValueError(f"La colonna target '{TARGET_COL}' non esiste nel dataset.")

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df[df[TARGET_COL].notna()].copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

if df.empty:
    raise ValueError("Il dataset è vuoto dopo la pulizia del target.")

print("\nDistribuzione target:")
display(
    df[TARGET_COL]
    .value_counts(dropna=False)
    .rename_axis("classe")
    .reset_index(name="conteggio")
)


# =========================================================
# 3. COSTRUZIONE FEATURE SET
# =========================================================

exclude_cols = [TARGET_COL]
if ID_COL is not None and ID_COL in df.columns:
    exclude_cols.append(ID_COL)

candidate_features = [c for c in df.columns if c not in exclude_cols]
candidate_features = [c for c in candidate_features if df[c].notna().sum() > 0]

X = df[candidate_features].copy()
y = df[TARGET_COL].copy()

for col in X.columns:
    X[col] = soft_convert_numeric(X[col])

feature_columns = X.columns.tolist()
numeric_cols, categorical_cols = build_feature_lists(X, MIN_UNIQUE_FOR_NUMERIC)

print("\nNumero feature candidate:", len(candidate_features))
print("Numero feature numeriche:", len(numeric_cols))
print("Numero feature categoriche:", len(categorical_cols))

print("\nPrime feature numeriche:")
print(numeric_cols[:20])

print("\nPrime feature categoriche:")
print(categorical_cols[:20])


# =========================================================
# 4. ANALISI DESCRITTIVA CLIENTI TAPPETO = 1
# =========================================================

df_pos = df[df[TARGET_COL] == 1].copy()
df_neg = df[df[TARGET_COL] == 0].copy()

print("\n" + "=" * 90)
print("PROFILO DESCRITTIVO CLIENTI CON TAPPETO = 1")
print("=" * 90)
print("Numero clienti tappeto=1:", len(df_pos))
print("Numero clienti tappeto=0:", len(df_neg))

# profilo numerico
numeric_profile_rows = []

for col in numeric_cols:
    pos_mean = pd.to_numeric(df_pos[col], errors="coerce").mean()
    neg_mean = pd.to_numeric(df_neg[col], errors="coerce").mean() if len(df_neg) > 0 else np.nan
    pos_median = pd.to_numeric(df_pos[col], errors="coerce").median()
    neg_median = pd.to_numeric(df_neg[col], errors="coerce").median() if len(df_neg) > 0 else np.nan
    diff_mean = pos_mean - neg_mean if pd.notna(neg_mean) else np.nan

    numeric_profile_rows.append({
        "variable": col,
        "mean_tappeto_1": pos_mean,
        "median_tappeto_1": pos_median,
        "mean_tappeto_0": neg_mean,
        "median_tappeto_0": neg_median,
        "diff_mean": diff_mean
    })

numeric_profile_df = pd.DataFrame(numeric_profile_rows)

if not numeric_profile_df.empty:
    numeric_profile_df["abs_diff_mean"] = numeric_profile_df["diff_mean"].abs()
    numeric_profile_df = numeric_profile_df.sort_values("abs_diff_mean", ascending=False).reset_index(drop=True)
    print("\nTop differenze variabili numeriche:")
    display(numeric_profile_df.drop(columns=["abs_diff_mean"]).head(20))
else:
    print("\nNessuna variabile numerica utile trovata.")

# profilo categorico
categorical_profile_all = []

for col in categorical_cols:
    tmp = (
        df.groupby([col, TARGET_COL], dropna=False)
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
    display(categorical_profile_df.head(40))
else:
    categorical_profile_df = pd.DataFrame()
    print("\nNessuna variabile categorica utile trovata.")


# =========================================================
# 5. MUTUAL INFORMATION
# =========================================================

X_mi = X.copy()
X_mi = X_mi.select_dtypes(exclude=["datetime64[ns]", "datetime64"])

for col in X_mi.columns:
    if not pd.api.types.is_numeric_dtype(X_mi[col]):
        X_mi[col] = X_mi[col].fillna("MISSING").astype(str).astype("category").cat.codes
    else:
        X_mi[col] = pd.to_numeric(X_mi[col], errors="coerce")
        X_mi[col] = X_mi[col].fillna(X_mi[col].median())

try:
    mi_scores = mutual_info_classif(X_mi, y, discrete_features="auto", random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({
        "variable": X_mi.columns,
        "mutual_info_score": mi_scores
    }).sort_values("mutual_info_score", ascending=False).reset_index(drop=True)

    print("\nTop variabili per Mutual Information:")
    display(mi_df.head(20))

except Exception as e:
    print("\nMutual Information non calcolabile:", e)
    mi_df = pd.DataFrame(columns=["variable", "mutual_info_score"])


# =========================================================
# 6. MODELLO PREDITTIVO
# =========================================================

if y.nunique() < 2:
    print("\n" + "=" * 90)
    print("ATTENZIONE: IL TARGET HA UNA SOLA CLASSE")
    print("Non è possibile allenare LogisticRegression.")
    print("Viene prodotta solo l'analisi descrittiva del profilo.")
    print("=" * 90)

    coef_df = pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
    df_scored = df.copy()
    top_clients = df.copy()

else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

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

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 90)
    print("VALUTAZIONE MODELLO")
    print("=" * 90)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.4f}")
    except Exception as e:
        print("ROC AUC non calcolabile:", e)

    coef_df = extract_model_coefficients(clf)

    if not coef_df.empty:
        print("\nTop coefficienti del modello:")
        display(coef_df.head(20))

    # scoring su tutto il dataset
    df_scored = df.copy()
    df_scored["prob_tappeto_1"] = clf.predict_proba(X)[:, 1]
    df_scored["pred_tappeto"] = clf.predict(X)

    top_clients = df_scored.sort_values("prob_tappeto_1", ascending=False).reset_index(drop=True)

    print("\nTop clienti per probabilità tappeto:")
    display(top_clients.head(20))

    # salva bundle modello
    model_bundle = {
        "pipeline": clf,
        "feature_columns": feature_columns,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": TARGET_COL,
        "id_col": ID_COL
    }

    model_path = os.path.join(OUTPUT_DIR, "modello_cluster_tappeto.joblib")
    joblib.dump(model_bundle, model_path)
    print(f"\nModello salvato in: {model_path}")


# =========================================================
# 7. SALVATAGGIO OUTPUT
# =========================================================

numeric_profile_path = os.path.join(OUTPUT_DIR, "profilo_numerico_tappeto.xlsx")
categorical_profile_path = os.path.join(OUTPUT_DIR, "profilo_categorico_tappeto.xlsx")
mi_path = os.path.join(OUTPUT_DIR, "mutual_info_tappeto.xlsx")
coef_path = os.path.join(OUTPUT_DIR, "feature_importance_modello.xlsx")
scored_path = os.path.join(OUTPUT_DIR, "dataset_scored_tappeto.xlsx")
top_clients_path = os.path.join(OUTPUT_DIR, "top_clienti_prob_tappeto.xlsx")
report_path = os.path.join(OUTPUT_DIR, "report_cluster_tappeto.xlsx")

if not numeric_profile_df.empty:
    numeric_profile_df.to_excel(numeric_profile_path, index=False)

if not categorical_profile_df.empty:
    categorical_profile_df.to_excel(categorical_profile_path, index=False)

if not mi_df.empty:
    mi_df.to_excel(mi_path, index=False)

if not coef_df.empty:
    coef_df.to_excel(coef_path, index=False)

if "prob_tappeto_1" in df_scored.columns:
    df_scored.to_excel(scored_path, index=False)
    top_clients.to_excel(top_clients_path, index=False)

save_excel_report(
    report_path,
    {
        "profilo_numerico": numeric_profile_df,
        "profilo_categorico": categorical_profile_df,
        "mutual_info": mi_df,
        "feature_model": coef_df,
        "dataset_scored": df_scored if "prob_tappeto_1" in df_scored.columns else df.head(500),
        "top_500_clienti": top_clients.head(500) if "prob_tappeto_1" in top_clients.columns else df.head(500)
    }
)

print("\nOutput salvati nella cartella:", OUTPUT_DIR)


# =========================================================
# 8. ESEMPIO DI RIUTILIZZO SU NUOVA LISTA NDG
# =========================================================

# Decommenta queste righe quando vuoi applicare il modello a un nuovo file:
#
# model_path = os.path.join(OUTPUT_DIR, "modello_cluster_tappeto.joblib")
# applica_modello_a_nuovo_file(
#     model_path=model_path,
#     new_file_path=r"nuova_lista_ndg.xlsx",
#     output_file_path=os.path.join(OUTPUT_DIR, "nuova_lista_ndg_scored.xlsx"),
#     sheet_name=0
# )
