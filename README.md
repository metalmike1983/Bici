# =========================================================
# APPLY SAVED MODEL TO NEW NDG LIST
# =========================================================

import pandas as pd
import joblib
import os

# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------
MODEL_PATH = r"C:\percorso\model_bundle.joblib"
INPUT_FILE = r"C:\percorso\nuova_lista_ndg.xlsx"
OUTPUT_FILE = r"C:\percorso\nuova_lista_ndg_scored.xlsx"

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]

scaler = bundle.get("scaler", None)
encoder = bundle.get("encoder", None)

# ---------------------------------------------------------
# LOAD NEW DATA
# ---------------------------------------------------------
df = pd.read_excel(INPUT_FILE)

# ---------------------------------------------------------
# ENSURE SAME FEATURES AS TRAINING
# ---------------------------------------------------------
for col in features:
    if col not in df.columns:
        df[col] = 0

X = df[features]

# ---------------------------------------------------------
# OPTIONAL TRANSFORMATIONS
# ---------------------------------------------------------
if encoder is not None:
    X = encoder.transform(X)

if scaler is not None:
    X = scaler.transform(X)

# ---------------------------------------------------------
# SCORING
# ---------------------------------------------------------
df["prob_tappeto_1"] = model.predict_proba(X)[:, 1]
df["prediction"] = model.predict(X)

# ---------------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------------
df.to_excel(OUTPUT_FILE, index=False)

print("Scoring completato")
print("Output salvato in:", OUTPUT_FILE)
