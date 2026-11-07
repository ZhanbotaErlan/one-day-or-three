from pathlib import Path

import pandas as pd


RAW_FOLDER = Path("data/raw")

kase_files = list(RAW_FOLDER.glob("kase_index.xls*"))

if not kase_files:
    raise FileNotFoundError(
        "No KASE file found. Put kase_index.xls or "
        "kase_index.xlsx inside data/raw."
    )

kase_file = kase_files[0]

print(f"KASE file: {kase_file.resolve()}")

excel_file = pd.ExcelFile(kase_file)

print("\nSheet names:")
print(excel_file.sheet_names)


for sheet_name in excel_file.sheet_names:
    print(f"\n--- Sheet: {sheet_name} ---")

    preview = pd.read_excel(
        kase_file,
        sheet_name=sheet_name,
        header=None,
        nrows=15,
    )

    print(preview.to_string(index=False, header=False))