from pathlib import Path

import pandas as pd


EVENTS_FILE = Path("data/processed/sp500_selloff_events.csv")
HSI_FILE = Path("data/raw/hsi.csv")

OUTPUT_FOLDER = Path("data/processed")
OUTPUT_FILE = OUTPUT_FOLDER / "hsi_event_responses.csv"


# Read the event list and Hang Seng Index data
events = pd.read_csv(
    EVENTS_FILE,
    parse_dates=["date"],
)

hsi = pd.read_csv(
    HSI_FILE,
    parse_dates=["date"],
)

events = events.sort_values("date").reset_index(drop=True)
hsi = hsi.sort_values("date").reset_index(drop=True)


matched_rows = []
skipped_events = []


for _, event in events.iterrows():
    event_date = event["date"]

    # Latest Hong Kong close available before the U.S. event session
    baseline = hsi[
        hsi["date"] <= event_date
    ].tail(1)

    # First three Hong Kong trading sessions after the U.S. event
    future_sessions = hsi[
        hsi["date"] > event_date
    ].head(3)

    if len(baseline) != 1 or len(future_sessions) < 3:
        skipped_events.append(event_date)
        continue

    baseline_row = baseline.iloc[0]
    day_1 = future_sessions.iloc[0]
    day_2 = future_sessions.iloc[1]
    day_3 = future_sessions.iloc[2]

    baseline_close = baseline_row["close"]

    matched_rows.append(
        {
            "us_event_date": event_date,
            "sp500_return_percent": event["return_percent"],

            "hsi_baseline_date": baseline_row["date"],
            "hsi_baseline_close": baseline_close,

            "hsi_day1_date": day_1["date"],
            "hsi_day1_close": day_1["close"],
            "hsi_return_day1": day_1["close"] / baseline_close - 1,

            "hsi_day2_date": day_2["date"],
            "hsi_day2_close": day_2["close"],
            "hsi_return_day2": day_2["close"] / baseline_close - 1,

            "hsi_day3_date": day_3["date"],
            "hsi_day3_close": day_3["close"],
            "hsi_return_day3": day_3["close"] / baseline_close - 1,
        }
    )


matched = pd.DataFrame(matched_rows)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
matched.to_csv(OUTPUT_FILE, index=False)


print(f"Original U.S. events: {len(events)}")
print(f"Successfully matched events: {len(matched)}")
print(f"Skipped events: {len(skipped_events)}")


print("\nFirst 10 date matches:")

print(
    matched[
        [
            "us_event_date",
            "hsi_baseline_date",
            "hsi_day1_date",
            "hsi_day2_date",
            "hsi_day3_date",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\nHang Seng cumulative returns after U.S. sell-offs:")

for day in [1, 2, 3]:
    column = f"hsi_return_day{day}"
    returns = matched[column]

    mean_return = returns.mean()
    median_return = returns.median()
    negative_share = (returns < 0).mean()

    print(
        f"Day {day}: "
        f"mean = {mean_return:.3%}, "
        f"median = {median_return:.3%}, "
        f"negative share = {negative_share:.1%}"
    )


print(f"\nResults saved to: {OUTPUT_FILE.resolve()}")
