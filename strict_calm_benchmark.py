from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

REPETITIONS = 10_000
BLOCK_LENGTHS = [20, 10, 40]
SEED = 42


# Read the three markets and both event samples.
sp500 = pd.read_csv(
    DATA / "raw/sp500.csv",
    parse_dates=["date"],
)

hsi = pd.read_csv(
    DATA / "raw/hsi.csv",
    parse_dates=["date"],
)

kase = pd.read_csv(
    DATA / "processed/kase_clean.csv",
    parse_dates=["date"],
)

strict_events = pd.read_csv(
    DATA / "processed/market_comparison_strict.csv",
    parse_dates=["us_event_date"],
)

all_events = pd.read_csv(
    DATA / "processed/market_comparison.csv",
    parse_dates=["us_event_date", "window_start", "window_end"],
)


for market in [sp500, hsi, kase]:
    market.sort_values("date", inplace=True)
    market.reset_index(drop=True, inplace=True)

    if market["date"].duplicated().any():
        raise ValueError("Duplicate dates in market data.")

    if market[["date", "close"]].isna().any().any():
        raise ValueError("Missing market dates or closing values.")

    if (market["close"] <= 0).any():
        raise ValueError("Non-positive closing values.")


sp500["daily_return"] = sp500["close"].pct_change(
    fill_method=None
)

us_dates = sp500["date"]


# Return a market window only when exactly one U.S.
# session lies between the baseline and local Day 1.
def market_window(market, event_date):
    next_position = int(
        market["date"].searchsorted(event_date, side="right")
    )

    if next_position == 0 or next_position + 2 >= len(market):
        return None

    baseline = market.iloc[next_position - 1]
    day1 = market.iloc[next_position]
    day3 = market.iloc[next_position + 2]

    first_us = int(
        us_dates.searchsorted(baseline["date"], side="left")
    )

    after_last_us = int(
        us_dates.searchsorted(day1["date"], side="left")
    )

    if after_last_us - first_us != 1:
        return None

    return {
        "start": baseline["date"],
        "end": day3["date"],
        "return_day1": day1["close"] / baseline["close"] - 1,
        "return_day3": day3["close"] / baseline["close"] - 1,
    }


# Keep every U.S. trading date in sequence, including
# dates that are ineligible for either comparison group.
rows = []

for event_date in us_dates:
    hong_kong = market_window(hsi, event_date)
    kazakhstan = market_window(kase, event_date)

    row = {
        "date": event_date,
        "eligible": False,
        "window_start": pd.NaT,
        "window_end": pd.NaT,
        "gap_day1": np.nan,
        "gap_day3": np.nan,
    }

    if hong_kong is not None and kazakhstan is not None:
        row.update(
            {
                "eligible": True,
                "window_start": min(
                    hong_kong["start"], kazakhstan["start"]
                ),
                "window_end": max(
                    hong_kong["end"], kazakhstan["end"]
                ),
                "gap_day1": (
                    kazakhstan["return_day1"]
                    - hong_kong["return_day1"]
                ),
                "gap_day3": (
                    kazakhstan["return_day3"]
                    - hong_kong["return_day3"]
                ),
            }
        )

    rows.append(row)


panel = pd.DataFrame(rows)
panel["year"] = panel["date"].dt.year
panel["sp500_return"] = sp500["daily_return"]

panel["selloff"] = (
    panel["eligible"]
    & (panel["sp500_return"] <= -0.02)
)


# Preserve the earlier exclusion of three U.S. sessions
# on either side of every raw sell-off.
near_selloff = np.zeros(len(panel), dtype=bool)

positions = np.flatnonzero(
    panel["sp500_return"].to_numpy() <= -0.02
)

for position in positions:
    start = max(0, position - 3)
    end = min(len(panel), position + 4)
    near_selloff[start:end] = True


# Also remove calm-day windows that overlap any raw
# sell-off observation window, including long holidays.
overlaps_selloff = np.zeros(len(panel), dtype=bool)

for event in all_events.itertuples():
    overlaps_selloff |= (
        (panel["window_start"] <= event.window_end)
        & (panel["window_end"] >= event.window_start)
    ).to_numpy()


panel["calm"] = (
    panel["eligible"]
    & (panel["sp500_return"].abs() <= 0.005)
    & (~near_selloff)
    & (~overlaps_selloff)
)


# Confirm that the event sample reproduces the audit.
actual_dates = set(panel.loc[panel["selloff"], "date"])
expected_dates = set(strict_events["us_event_date"])

if actual_dates != expected_dates:
    raise ValueError(
        "Event dates differ from market_comparison_strict.csv."
    )

check = panel.loc[
    panel["selloff"], ["date", "gap_day1", "gap_day3"]
].set_index("date")

reference = strict_events.set_index("us_event_date")

for column in ["gap_day1", "gap_day3"]:
    if not np.allclose(
        check[column],
        reference.loc[check.index, column],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(f"Return mismatch in {column}.")


# Use years represented in the strict sell-off sample.
event_years = sorted(
    panel.loc[panel["selloff"], "year"].unique()
)

annual_values = []

for year in event_years:
    sample = panel[panel["year"] == year]

    event_mask = sample["selloff"].to_numpy()
    calm_mask = sample["calm"].to_numpy()

    if not calm_mask.any():
        raise ValueError(f"No eligible calm days in {year}.")

    gap1 = sample["gap_day1"].to_numpy()
    gap3 = sample["gap_day3"].to_numpy()

    # Counts and return sums allow both groups and their
    # annual weights to be recalculated in every resample.
    values = np.column_stack(
        [
            event_mask.astype(float),
            calm_mask.astype(float),
            np.where(event_mask, gap1, 0.0),
            np.where(event_mask, gap3, 0.0),
            np.where(calm_mask, gap1, 0.0),
            np.where(calm_mask, gap3, 0.0),
        ]
    )

    annual_values.append(values)


# Input dimensions are repetitions x years x six statistics.
def calculate_estimates(totals):
    event_counts = totals[:, :, 0]
    calm_counts = totals[:, :, 1]
    total_events = event_counts.sum(axis=1)

    valid = (
        (total_events > 0)
        & ~((event_counts > 0) & (calm_counts == 0)).any(axis=1)
    )

    calm_means = np.divide(
        totals[:, :, 4:6],
        calm_counts[:, :, None],
        out=np.zeros_like(totals[:, :, 4:6]),
        where=calm_counts[:, :, None] > 0,
    )

    denominator = np.where(
        total_events > 0, total_events, np.nan
    )[:, None]

    event_means = totals[:, :, 2:4].sum(axis=1) / denominator

    weighted_calm = (
        calm_means * event_counts[:, :, None]
    ).sum(axis=1) / denominator

    excess = event_means - weighted_calm

    return event_means, weighted_calm, excess, valid


original_totals = np.array(
    [values.sum(axis=0) for values in annual_values]
)[None, :, :]

event_mean, calm_mean, excess, valid = calculate_estimates(
    original_totals
)

used_calm = panel["calm"] & panel["year"].isin(event_years)

print(f"Calendar-filtered sell-off events: {panel['selloff'].sum()}")
print(f"Calm observations in comparison years: {used_calm.sum()}")

for position, day in enumerate([1, 3]):
    print(f"\nDay {day}:")
    print(f"Sell-off gap: {event_mean[0, position] * 100:.3f} pp")
    print(f"Weighted calm gap: {calm_mean[0, position] * 100:.3f} pp")
    print(f"Excess gap: {excess[0, position] * 100:.3f} pp")


# Draw circular blocks within each calendar year.
# Block sums make this substantially faster than rebuilding
# a DataFrame for every bootstrap repetition.
def resample_year(values, block_length, rng):
    number_of_days = len(values)
    full_blocks, remainder = divmod(number_of_days, block_length)

    doubled = np.concatenate([values, values], axis=0)
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1])), doubled.cumsum(axis=0)]
    )

    starts = rng.integers(
        0,
        number_of_days,
        size=(REPETITIONS, full_blocks),
    )

    totals = (
        cumulative[starts + block_length] - cumulative[starts]
    ).sum(axis=1)

    if remainder:
        tail_starts = rng.integers(
            0, number_of_days, size=REPETITIONS
        )

        totals += (
            cumulative[tail_starts + remainder]
            - cumulative[tail_starts]
        )

    return totals


summary_rows = []

for block_length in BLOCK_LENGTHS:
    rng = np.random.default_rng(SEED + block_length)

    bootstrap_totals = np.stack(
        [
            resample_year(values, block_length, rng)
            for values in annual_values
        ],
        axis=1,
    )

    _, _, bootstrap_excess, valid = calculate_estimates(
        bootstrap_totals
    )

    print(
        f"\nBlock length {block_length}: "
        f"{valid.sum()}/{REPETITIONS} valid repetitions"
    )

    if valid.mean() < 0.90:
        raise ValueError(
            "Too many resamples lack usable comparison groups. "
            "Send this output before interpreting the intervals."
        )

    bootstrap_excess = bootstrap_excess[valid]

    point_estimates = [
        excess[0, 0],
        excess[0, 1],
        excess[0, 1] - excess[0, 0],
    ]

    distributions = [
        bootstrap_excess[:, 0],
        bootstrap_excess[:, 1],
        bootstrap_excess[:, 1] - bootstrap_excess[:, 0],
    ]

    labels = [
        "Day 1 excess",
        "Day 3 excess",
        "Change in excess, Day 1 to Day 3",
    ]

    for label, estimate, distribution in zip(
        labels, point_estimates, distributions
    ):
        lower, upper = np.quantile(distribution, [0.025, 0.975])

        print(
            f"{label}: {estimate * 100:.3f} pp "
            f"[{lower * 100:.3f}, {upper * 100:.3f}]"
        )

        summary_rows.append(
            {
                "block_length": block_length,
                "metric": label,
                "estimate_pp": estimate * 100,
                "ci_lower_pp": lower * 100,
                "ci_upper_pp": upper * 100,
                "valid_repetitions": int(valid.sum()),
            }
        )


RESULTS.mkdir(parents=True, exist_ok=True)

panel.to_csv(
    DATA / "processed/strict_benchmark_panel.csv",
    index=False,
)

pd.DataFrame(summary_rows).to_csv(
    RESULTS / "strict_calm_benchmark.csv",
    index=False,
)

print("\nFiltered benchmark and bootstrap results saved.")