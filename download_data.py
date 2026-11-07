from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas_datareader import data as web


START_DATE = "2017-01-01"
END_DATE = "2026-01-01"

DATA_FOLDER = Path("data/raw")
DATA_FOLDER.mkdir(parents=True, exist_ok=True)


# Download S&P 500 daily closing values from FRED
sp500 = web.DataReader(
    "SP500",
    "fred",
    START_DATE,
    END_DATE,
)

sp500 = (
    sp500
    .dropna()
    .rename(columns={"SP500": "close"})
    .reset_index()
    .rename(columns={"DATE": "date"})
)

sp500["date"] = pd.to_datetime(sp500["date"])
sp500.to_csv(DATA_FOLDER / "sp500.csv", index=False)


# Download Hang Seng Index daily prices
hsi = yf.Ticker("^HSI").history(
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
)

hsi = (
    hsi[["Open", "Close"]]
    .reset_index()
    .rename(
        columns={
            "Date": "date",
            "Open": "open",
            "Close": "close",
        }
    )
)

hsi["date"] = pd.to_datetime(hsi["date"]).dt.tz_localize(None)
hsi.to_csv(DATA_FOLDER / "hsi.csv", index=False)


# Display the results
print("\nS&P 500")
print(sp500.head())
print(f"Rows: {len(sp500)}")
print(f"Date range: {sp500['date'].min()} to {sp500['date'].max()}")

print("\nHang Seng Index")
print(hsi.head())
print(f"Rows: {len(hsi)}")
print(f"Date range: {hsi['date'].min()} to {hsi['date'].max()}")

print("\nFiles saved in:", DATA_FOLDER.resolve())