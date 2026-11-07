from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw/sp500.csv")
OUTPUT_FOLDER = Path("data/processed")
OUTPUT_FILE = OUTPUT_FOLDER / "sp500_selloff_events.csv"

SELL_OFF_THRESHOLD = -0.02


# Read and arrange the S&P 500 data
sp500 = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"],
)

sp500 = sp500.sort_values("date").reset_index(drop=True)


# Calculate each trading day's close-to-close return
sp500["daily_return"] = sp500["close"].pct_change()


# Select days when the S&P 500 fell by at least 2%
events = sp500[
    sp500["daily_return"] <= SELL_OFF_THRESHOLD
].copy()

events["return_percent"] = events["daily_return"] * 100


# Save the event list
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

events[
    ["date", "close", "daily_return", "return_percent"]
].to_csv(
    OUTPUT_FILE,
    index=False,
)


# Display the result
print(f"Sell-off threshold: {SELL_OFF_THRESHOLD:.0%}")
print(f"Number of events: {len(events)}")

print("\nFirst 10 events:")
print(
    events[
        ["date", "close", "return_percent"]
    ].head(10).to_string(index=False)
)

print("\n10 largest declines:")
print(
    events[
        ["date", "close", "return_percent"]
    ]
    .sort_values("return_percent")
    .head(10)
    .to_string(index=False)
)

print(f"\nEvent list saved to: {OUTPUT_FILE.resolve()}")