# Se non hai tqdm:
# !pip install -q tqdm

import re
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from openpyxl import load_workbook
from tqdm.auto import tqdm

# =========================
# CONFIG
# =========================
FOLDER = Path(r"C:\Users\TUO_UTENTE\OneDrive\GISMONDI Cesidio\Dossier_Top\storico_xml_scaricati")
OUT_XLSX = "estrazione_xml_ndg.xlsx"
SHEET_NAME = "data"
CHUNK_SIZE = 100

# RESUME: se True, salta i file già presenti nell'output Excel (colonna "file")
RESUME = True

# =========================
# HELPERS
# =========================
def extract_ndg_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_", 1)[0]

# ---- Se vuoi SOLO la parte numerica dell'NDG, usa questa al posto di sopra:
# def extract_ndg_from_filename(filename: str) -> str:
#     stem = Path(filename).stem
#     prefix = stem.split("_", 1)[0]
#     m = re.search(r"\d+", prefix)
#     return m.group() if m else prefix

def tag_matches(elem_tag: str, wanted: str) -> bool:
    local = elem_tag.split("}", 1)[-1]
    return local.lower() == wanted.lower()

def find_first_text(root: ET.Element, tag_name: str) -> str | None:
    for el in root.iter():
        if tag_matches(el.tag, tag_name):
            if el.text and el.text.strip():
                return el.text.strip()
    return None

def extract_fields(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ndg = extract_ndg_from_filename(xml_path.name)

    email = find_first_text(root, "Email")
    pec = find_first_text(root, "CertifiedEmail")
    company = find_first_text(root, "CompanyName")

    if company is None:
        company = (
            find_first_text(root, "Denominazione")
            or find_first_text(root, "BusinessName")
            or find_first_text(root, "Company")
        )

    return {
        "ndg": ndg,
        "file": xml_path.name,
        "email": email,
        "pec": pec,
        "denominazione": company,
    }

def append_df_to_excel(excel_path: Path, df: pd.DataFrame, sheet_name: str = "data"):
    if df.empty:
        return

    if not excel_path.exists():
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        return

    wb = load_workbook(excel_path)

    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        startrow = ws.max_row  # include header
    else:
        wb.create_sheet(sheet_name)
        startrow = 0

    wb.save(excel_path)

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
            header=(startrow == 0),
            startrow=startrow,
        )

def load_already_processed_files(excel_path: Path, sheet_name: str) -> set[str]:
    """
    Legge la colonna 'file' dall'Excel e costruisce un set per modalità RESUME.
    Se non esiste/errore/sheet mancante -> set vuoto.
    """
    if not excel_path.exists():
        return set()
    try:
        existing = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=["file"])
        existing = existing.dropna()
        return set(existing["file"].astype(str).tolist())
    except Exception:
        return set()

# =========================
# RUN
# =========================
folder = FOLDER
if not folder.exists():
    raise FileNotFoundError(f"Cartella non trovata: {folder}")

excel_path = folder / OUT_XLSX

# Resume set
already_done = load_already_processed_files(excel_path, SHEET_NAME) if RESUME else set()

all_xml_files = list(folder.rglob("*.xml"))

buffer_rows = []
errors = []
processed = 0          # record effettivamente estratti (nuovi)
skipped_resume = 0     # saltati perché già in output
saved_batches = 0

pbar = tqdm(all_xml_files, desc="Scansione XML", unit="file")

for xml_file in pbar:
    ndg_prefix = extract_ndg_from_filename(xml_file.name)

    # filtro: prende solo i file dove "ndg" compare nel prefisso (prima del primo underscore)
    if "ndg" not in ndg_prefix.lower():
        continue

    # RESUME: se il file è già stato scritto nell'output, lo salto
    if RESUME and (xml_file.name in already_done):
        skipped_resume += 1
        pbar.set_postfix(processed=processed, skipped=skipped_resume, batches=saved_batches, errors=len(errors))
        continue

    try:
        row = extract_fields(xml_file)
        buffer_rows.append(row)
        processed += 1
        if RESUME:
            already_done.add(xml_file.name)  # evita duplicazioni nella stessa run
    except Exception as e:
        errors.append({"file": xml_file.name, "error": repr(e)})

    # salva ogni CHUNK_SIZE record nuovi
    if processed > 0 and processed % CHUNK_SIZE == 0:
        df_chunk = pd.DataFrame(buffer_rows).sort_values(["ndg", "file"]).reset_index(drop=True)
        append_df_to_excel(excel_path, df_chunk, sheet_name=SHEET_NAME)
        saved_batches += 1
        buffer_rows.clear()

    pbar.set_postfix(processed=processed, skipped=skipped_resume, batches=saved_batches, errors=len(errors))

# salva residui
if buffer_rows:
    df_chunk = pd.DataFrame(buffer_rows).sort_values(["ndg", "file"]).reset_index(drop=True)
    append_df_to_excel(excel_path, df_chunk, sheet_name=SHEET_NAME)

# salva errori
if errors:
    err_df = pd.DataFrame(errors)
    err_path = folder / "estrazione_xml_ndg_ERRORI.xlsx"
    with pd.ExcelWriter(err_path, engine="openpyxl") as writer:
        err_df.to_excel(writer, sheet_name="errori", index=False)
    print(f"ATTENZIONE: {len(errors)} errori. Vedi: {err_path}")

print(f"Fatto. Nuovi record estratti: {processed}. Saltati (resume): {skipped_resume}. Output: {excel_path}")
