from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw/kase_index.xlsx")
OUTPUT_FOLDER = Path("data/processed")
OUTPUT_FILE = OUTPUT_FOLDER / "kase_clean.csv"


# Row 1 contains the file title.
# Row 2 contains the actual column names.
raw = pd.read_excel(
    INPUT_FILE,
    sheet_name=0,
    header=1,
)

raw.columns = raw.columns.astype(str).str.strip()


required_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume, KZT m",
    "Volume, USD th",
]

missing_columns = [
    column
    for column in required_columns
    if column not in raw.columns
]

if missing_columns:
    raise KeyError(
        f"Missing expected columns: {missing_columns}"
    )


kase = raw[required_columns].copy()

kase = kase.rename(
    columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume, KZT m": "volume_kzt_m",
        "Volume, USD th": "volume_usd_th",
    }
)


# Convert dates and prices into consistent formats
kase["date"] = pd.to_datetime(
    kase["date"],
    format="%d.%m.%y",
    errors="coerce",
)

numeric_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume_kzt_m",
    "volume_usd_th",
]

for column in numeric_columns:
    kase[column] = pd.to_numeric(
        kase[column],
        errors="coerce",
    )


# Remove invalid rows and arrange chronologically
kase = (
    kase
    .dropna(subset=["date", "close"])
    .sort_values("date")
    .drop_duplicates(subset=["date"], keep="last")
    .reset_index(drop=True)
)


# Calculate daily close-to-close returns for data checks
kase["daily_return"] = kase["close"].pct_change()

zero_return_days = (
    kase["daily_return"] == 0
).sum()


# Save cleaned data
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
kase.to_csv(OUTPUT_FILE, index=False)


print("KASE cleaned data")
print(f"Rows: {len(kase)}")
print(
    f"Date range: {kase['date'].min()} "
    f"to {kase['date'].max()}"
)
print(f"Duplicate dates: {kase['date'].duplicated().sum()}")
print(f"Zero-return days: {zero_return_days}")

print("\nMissing values:")
print(kase.isna().sum())

print("\nFirst 5 rows:")
print(kase.head().to_string(index=False))

print("\nLast 5 rows:")
print(kase.tail().to_string(index=False))

print(f"\nCleaned file saved to: {OUTPUT_FILE.resolve()}")