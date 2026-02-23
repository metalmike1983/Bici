import re
from pathlib import Path as _Path
import xml.etree.ElementTree as ET

import pandas as pd
from openpyxl import load_workbook


# =========================
# CONFIG
# =========================
FOLDER = r"C:\Users\TUO_UTENTE\OneDrive\GISMONDI Cesidio\Dossier_Top\storico_xml_scaricati"
OUT_XLSX = "estrazione_xml_ndg.xlsx"
SHEET_NAME = "data"
CHUNK_SIZE = 100


def extract_ndg_from_filename(filename: str) -> str:
    """
    NDG preso dal nome file: parte prima del primo underscore.
    Esempi:
      'NDG308093363_10485960966.xml' -> 'NDG308093363'
      '308093363_10485960966.xml'    -> '308093363'
    Se vuoi solo la parte numerica, vedi commento più sotto.
    """
    stem = _Path(filename).stem
    return stem.split("_", 1)[0]


# ---- Se vuoi SOLO la parte numerica dell'NDG, sostituisci la funzione sopra con questa:
# def extract_ndg_from_filename(filename: str) -> str:
#     stem = _Path(filename).stem
#     prefix = stem.split("_", 1)[0]
#     m = re.search(r"\d+", prefix)
#     return m.group() if m else prefix


def tag_matches(elem_tag: str, wanted: str) -> bool:
    """Confronta il nome del tag ignorando namespace XML."""
    local = elem_tag.split("}", 1)[-1]  # '{ns}CompanyName' -> 'CompanyName'
    return local.lower() == wanted.lower()


def find_first_text(root: ET.Element, tag_name: str) -> str | None:
    """Trova la prima occorrenza del tag e ritorna il testo pulito (ignorando namespace)."""
    for el in root.iter():
        if tag_matches(el.tag, tag_name):
            if el.text and el.text.strip():
                return el.text.strip()
    return None


def extract_fields(xml_path: _Path) -> dict:
    """Estrae NDG (da filename), email, PEC, denominazione dall'XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ndg = extract_ndg_from_filename(xml_path.name)

    # Tag visti nei tuoi screenshot:
    email = find_first_text(root, "Email")
    pec = find_first_text(root, "CertifiedEmail")
    company = find_first_text(root, "CompanyName")

    # Fallback (se in alcuni file i tag differiscono)
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


def append_df_to_excel(excel_path: _Path, df: pd.DataFrame, sheet_name: str = "data"):
    """
    Accoda df in fondo al foglio sheet_name.
    Se il file non esiste, lo crea con header.
    """
    if df.empty:
        return

    if not excel_path.exists():
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        return

    wb = load_workbook(excel_path)

    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        startrow = ws.max_row  # include header, quindi parte dalla riga successiva
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


def main():
    folder = _Path(FOLDER)
    if not folder.exists():
        raise FileNotFoundError(f"Cartella non trovata: {folder}")

    excel_path = folder / OUT_XLSX

    buffer_rows = []
    errors = []
    processed = 0
    saved_batches = 0

    for xml_file in folder.rglob("*.xml"):
        ndg_prefix = extract_ndg_from_filename(xml_file.name)

        # filtro: prende solo i file dove "ndg" compare nel prefisso (prima del primo underscore)
        if "ndg" not in ndg_prefix.lower():
            continue

        try:
            buffer_rows.append(extract_fields(xml_file))
            processed += 1
        except Exception as e:
            errors.append({"file": xml_file.name, "error": repr(e)})

        # ogni CHUNK_SIZE -> salva e svuota buffer
        if processed > 0 and processed % CHUNK_SIZE == 0:
            df_chunk = pd.DataFrame(buffer_rows)
            df_chunk = df_chunk.sort_values(["ndg", "file"]).reset_index(drop=True)
            append_df_to_excel(excel_path, df_chunk, sheet_name=SHEET_NAME)

            saved_batches += 1
            print(f"Salvato batch {saved_batches} ({processed} record totali) -> {excel_path}")
            buffer_rows.clear()

    # salva eventuali residui
    if buffer_rows:
        df_chunk = pd.DataFrame(buffer_rows)
        df_chunk = df_chunk.sort_values(["ndg", "file"]).reset_index(drop=True)
        append_df_to_excel(excel_path, df_chunk, sheet_name=SHEET_NAME)
        print(f"Salvato batch finale (+{len(df_chunk)} record) -> {excel_path}")

    # salva errori (facoltativo)
    if errors:
        err_df = pd.DataFrame(errors)
        err_path = folder / "estrazione_xml_ndg_ERRORI.xlsx"
        with pd.ExcelWriter(err_path, engine="openpyxl") as writer:
            err_df.to_excel(writer, sheet_name="errori", index=False)
        print(f"ATTENZIONE: {len(errors)} errori. Vedi: {err_path}")

    print(f"Fatto. Totale record processati: {processed}. Output: {excel_path}")


if __name__ == "__main__":
    main()
