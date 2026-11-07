from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
RESULTS_FOLDER = PROJECT_FOLDER / "results"

BOOTSTRAP_REPETITIONS = 10_000
RANDOM_SEED = 42

METRICS = [
    "gap_day1",
    "gap_day3",
    "gap_change_day1_to_day3",
]


# Read existing data
sp500 = pd.read_csv(
    DATA_FOLDER / "raw/sp500.csv",
    parse_dates=["date"],
)

comparison = pd.read_csv(
    DATA_FOLDER / "processed/market_comparison.csv"
)

for column in comparison.columns:
    if column.endswith("_date") or column in [
        "window_start",
        "window_end",
    ]:
        comparison[column] = pd.to_datetime(
            comparison[column],
            errors="raise",
        )

sp500 = sp500.sort_values("date").reset_index(drop=True)

if sp500[["date", "close"]].isna().any().any():
    raise ValueError("Missing S&P 500 dates or closing values.")

if sp500["date"].duplicated().any():
    raise ValueError("Duplicate S&P 500 dates.")

if comparison["us_event_date"].duplicated().any():
    raise ValueError("Duplicate event dates.")

if comparison[METRICS].isna().any().any():
    raise ValueError("Missing return-gap values.")


us_dates = sp500["date"]

# Count U.S. trading dates from the local baseline date
# up to, but excluding, the local Day 1 date.
#
# Include the baseline date because the U.S. close occurs
# after that date's local-market close.
for market in ["hsi", "kase"]:
    baseline = comparison[f"{market}_baseline_date"]
    day1 = comparison[f"{market}_day1_date"]
    day2 = comparison[f"{market}_day2_date"]
    day3 = comparison[f"{market}_day3_date"]
    event = comparison["us_event_date"]

    dates_valid = (
        (baseline <= event)
        & (event < day1)
        & (day1 < day2)
        & (day2 < day3)
    )

    if not dates_valid.all():
        raise ValueError(
            f"Unexpected date order in {market.upper()} windows."
        )

    session_counts = []

    for baseline_date, day1_date in zip(baseline, day1):
        count = (
            (us_dates >= baseline_date)
            & (us_dates < day1_date)
        ).sum()

        session_counts.append(int(count))

    comparison[f"{market}_us_sessions"] = session_counts


# Both markets must meet the same one-U.S.-session rule
comparison["passes_calendar_filter"] = (
    (comparison["hsi_us_sessions"] == 1)
    & (comparison["kase_us_sessions"] == 1)
)

strict = comparison[
    comparison["passes_calendar_filter"]
].copy()

excluded = comparison[
    ~comparison["passes_calendar_filter"]
].copy()

if strict.empty:
    raise ValueError("No events passed the calendar filter.")


# Rebuild overlap clusters after filtering.
# Sort by window start to group intersecting intervals.
def assign_clusters(dataframe):
    dataframe = dataframe.sort_values(
        ["window_start", "window_end", "us_event_date"]
    ).copy()

    cluster_ids = []
    cluster_id = 0
    cluster_end = None

    for row in dataframe.itertuples():
        if cluster_end is None or row.window_start > cluster_end:
            cluster_id += 1
            cluster_end = row.window_end
        else:
            cluster_end = max(cluster_end, row.window_end)

        cluster_ids.append(cluster_id)

    dataframe["cluster_id"] = cluster_ids

    return dataframe.sort_values(
        "us_event_date"
    ).reset_index(drop=True)


strict = assign_clusters(strict)


print(f"Original matched events: {len(comparison)}")
print(f"Events passing calendar filter: {len(strict)}")
print(f"Excluded events: {len(excluded)}")
print(f"Filtered overlap clusters: {strict['cluster_id'].nunique()}")

print("\nExcluded event windows:")
print(
    excluded[
        [
            "us_event_date",
            "hsi_baseline_date",
            "hsi_day1_date",
            "hsi_us_sessions",
            "kase_baseline_date",
            "kase_day1_date",
            "kase_us_sessions",
        ]
    ].to_string(index=False)
)


# Sample whole clusters, keeping the paired market results.
groups = strict.groupby("cluster_id")
cluster_sums = groups[METRICS].sum().to_numpy()
cluster_counts = groups.size().to_numpy()

number_of_clusters = len(cluster_counts)

if number_of_clusters < 2:
    raise ValueError("Too few clusters for bootstrap inference.")

rng = np.random.default_rng(RANDOM_SEED)

sampled_indices = rng.integers(
    0,
    number_of_clusters,
    size=(BOOTSTRAP_REPETITIONS, number_of_clusters),
)

sampled_sums = cluster_sums[sampled_indices].sum(axis=1)
sampled_counts = cluster_counts[sampled_indices].sum(axis=1)

bootstrap_means = sampled_sums / sampled_counts[:, None]


print("\nCalendar-filtered results:")

summary_rows = []

for position, metric in enumerate(METRICS):
    estimate = strict[metric].mean()
    lower, upper = np.quantile(
        bootstrap_means[:, position],
        [0.025, 0.975],
    )

    print(
        f"{metric}: {estimate * 100:.3f} pp "
        f"[{lower * 100:.3f}, {upper * 100:.3f}]"
    )

    summary_rows.append(
        {
            "metric": metric,
            "events": len(strict),
            "clusters": number_of_clusters,
            "estimate_pp": estimate * 100,
            "ci_lower_pp": lower * 100,
            "ci_upper_pp": upper * 100,
        }
    )


# Save the audit trail and filtered analysis sample
comparison.to_csv(
    DATA_FOLDER / "processed/market_comparison_calendar_audit.csv",
    index=False,
)

strict.to_csv(
    DATA_FOLDER / "processed/market_comparison_strict.csv",
    index=False,
)

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

pd.DataFrame(summary_rows).to_csv(
    RESULTS_FOLDER / "calendar_audit_summary.csv",
    index=False,
)

print("\nCalendar audit and filtered results saved.")