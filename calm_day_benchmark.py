from pathlib import Path

import numpy as np
import pandas as pd


SP500_FILE = Path("data/raw/sp500.csv")
HSI_FILE = Path("data/raw/hsi.csv")
KASE_FILE = Path("data/processed/kase_clean.csv")

OUTPUT_FOLDER = Path("results")
OUTPUT_FILE = OUTPUT_FOLDER / "calm_day_benchmark.csv"

SELL_OFF_THRESHOLD = -0.02
CALM_LIMIT = 0.005
EXCLUSION_WINDOW = 3


sp500 = pd.read_csv(
    SP500_FILE,
    parse_dates=["date"],
)

hsi = pd.read_csv(
    HSI_FILE,
    parse_dates=["date"],
)

kase = pd.read_csv(
    KASE_FILE,
    parse_dates=["date"],
)


sp500 = sp500.sort_values("date").reset_index(drop=True)
hsi = hsi.sort_values("date").reset_index(drop=True)
kase = kase.sort_values("date").reset_index(drop=True)

sp500["daily_return"] = sp500["close"].pct_change()


# Identify sell-off days
selloff_mask = (
    sp500["daily_return"] <= SELL_OFF_THRESHOLD
)


# Exclude three U.S. trading sessions before and after
# every sell-off from the calm-day sample
near_selloff = np.zeros(
    len(sp500),
    dtype=bool,
)

selloff_positions = np.where(
    selloff_mask
)[0]

for position in selloff_positions:
    start = max(
        0,
        position - EXCLUSION_WINDOW,
    )

    end = min(
        len(sp500),
        position + EXCLUSION_WINDOW + 1,
    )

    near_selloff[start:end] = True


calm_mask = (
    sp500["daily_return"].abs() <= CALM_LIMIT
) & (~near_selloff)


selloff_days = sp500[
    selloff_mask
][
    ["date", "daily_return"]
].copy()

calm_days = sp500[
    calm_mask
][
    ["date", "daily_return"]
].copy()


def get_market_window(market, event_date):
    baseline = market[
        market["date"] <= event_date
    ].tail(1)

    future = market[
        market["date"] > event_date
    ].head(3)

    if len(baseline) != 1 or len(future) < 3:
        return None

    return baseline.iloc[0], future.reset_index(drop=True)


def build_responses(event_days):
    rows = []

    for event in event_days.itertuples():
        hsi_window = get_market_window(
            hsi,
            event.date,
        )

        kase_window = get_market_window(
            kase,
            event.date,
        )

        if hsi_window is None or kase_window is None:
            continue

        hsi_baseline, hsi_future = hsi_window
        kase_baseline, kase_future = kase_window

        hsi_day1_return = (
            hsi_future.iloc[0]["close"]
            / hsi_baseline["close"]
            - 1
        )

        hsi_day3_return = (
            hsi_future.iloc[2]["close"]
            / hsi_baseline["close"]
            - 1
        )

        kase_day1_return = (
            kase_future.iloc[0]["close"]
            / kase_baseline["close"]
            - 1
        )

        kase_day3_return = (
            kase_future.iloc[2]["close"]
            / kase_baseline["close"]
            - 1
        )

        rows.append(
            {
                "event_date": event.date,
                "year": event.date.year,
                "sp500_return": event.daily_return,

                "gap_day1":
                    kase_day1_return
                    - hsi_day1_return,

                "gap_day3":
                    kase_day3_return
                    - hsi_day3_return,
            }
        )

    return pd.DataFrame(rows)


selloff_responses = build_responses(
    selloff_days
)

calm_responses = build_responses(
    calm_days
)


selloff_by_year = (
    selloff_responses
    .groupby("year")
    .agg(
        selloff_events=("event_date", "count"),
        selloff_gap_day1=("gap_day1", "mean"),
        selloff_gap_day3=("gap_day3", "mean"),
    )
    .reset_index()
)


calm_by_year = (
    calm_responses
    .groupby("year")
    .agg(
        calm_events=("event_date", "count"),
        calm_gap_day1=("gap_day1", "mean"),
        calm_gap_day3=("gap_day3", "mean"),
    )
    .reset_index()
)


comparison = selloff_by_year.merge(
    calm_by_year,
    on="year",
    how="left",
)


# Weight calm-year averages using the distribution
# of sell-off events across years
comparison["selloff_weight"] = (
    comparison["selloff_events"]
    / comparison["selloff_events"].sum()
)


weighted_calm_gap_day1 = (
    comparison["calm_gap_day1"]
    * comparison["selloff_weight"]
).sum()

weighted_calm_gap_day3 = (
    comparison["calm_gap_day3"]
    * comparison["selloff_weight"]
).sum()


selloff_gap_day1 = (
    selloff_responses["gap_day1"].mean()
)

selloff_gap_day3 = (
    selloff_responses["gap_day3"].mean()
)


excess_gap_day1 = (
    selloff_gap_day1
    - weighted_calm_gap_day1
)

excess_gap_day3 = (
    selloff_gap_day3
    - weighted_calm_gap_day3
)


comparison["day1_difference"] = (
    comparison["selloff_gap_day1"]
    - comparison["calm_gap_day1"]
)

comparison["day3_difference"] = (
    comparison["selloff_gap_day3"]
    - comparison["calm_gap_day3"]
)


print(f"Sell-off days: {len(selloff_responses)}")
print(f"Eligible calm days: {len(calm_responses)}")


print("\nYear-level comparison:")

display = comparison.copy()

percentage_columns = [
    "selloff_gap_day1",
    "calm_gap_day1",
    "day1_difference",
    "selloff_gap_day3",
    "calm_gap_day3",
    "day3_difference",
]

for column in percentage_columns:
    display[column] = display[column] * 100

print(display.to_string(index=False))


print("\nSell-off weighted calm-day benchmark:")

print(
    f"Sell-off Day 1 gap: "
    f"{selloff_gap_day1 * 100:.3f} pp"
)

print(
    f"Weighted calm Day 1 gap: "
    f"{weighted_calm_gap_day1 * 100:.3f} pp"
)

print(
    f"Day 1 excess over calm benchmark: "
    f"{excess_gap_day1 * 100:.3f} pp"
)


print(
    f"\nSell-off Day 3 gap: "
    f"{selloff_gap_day3 * 100:.3f} pp"
)

print(
    f"Weighted calm Day 3 gap: "
    f"{weighted_calm_gap_day3 * 100:.3f} pp"
)

print(
    f"Day 3 excess over calm benchmark: "
    f"{excess_gap_day3 * 100:.3f} pp"
)


OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"\nResults saved to: "
    f"{OUTPUT_FILE.resolve()}"
)