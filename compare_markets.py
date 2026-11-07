from pathlib import Path

import pandas as pd


HSI_FILE = Path("data/processed/hsi_event_responses.csv")
KASE_FILE = Path("data/processed/kase_event_responses.csv")

OUTPUT_FOLDER = Path("data/processed")
OUTPUT_FILE = OUTPUT_FOLDER / "market_comparison.csv"


hsi = pd.read_csv(HSI_FILE)
kase = pd.read_csv(KASE_FILE)


# Convert all date columns into datetime values
for dataframe in [hsi, kase]:
    for column in dataframe.columns:
        if column.endswith("_date"):
            dataframe[column] = pd.to_datetime(
                dataframe[column]
            )


# Keep one copy of the S&P 500 return
kase = kase.drop(
    columns=["sp500_return_percent"]
)


# Match Hong Kong and Kazakhstan using the U.S. event date
comparison = hsi.merge(
    kase,
    on="us_event_date",
    how="inner",
    validate="one_to_one",
)


# Calculate the return gap at each horizon
for day in [1, 2, 3]:
    comparison[f"gap_day{day}"] = (
        comparison[f"kase_return_day{day}"]
        - comparison[f"hsi_return_day{day}"]
    )


# Main research statistic
comparison["gap_change_day1_to_day3"] = (
    comparison["gap_day3"]
    - comparison["gap_day1"]
)


# Define the full observation window for each event
comparison["window_start"] = comparison[
    [
        "hsi_baseline_date",
        "kase_baseline_date",
    ]
].min(axis=1)

comparison["window_end"] = comparison[
    [
        "hsi_day3_date",
        "kase_day3_date",
    ]
].max(axis=1)


comparison = comparison.sort_values(
    "us_event_date"
).reset_index(drop=True)


# Assign the same cluster number to overlapping event windows
cluster_ids = []
cluster_id = 0
cluster_end = None

for row in comparison.itertuples():
    if cluster_end is None or row.window_start > cluster_end:
        cluster_id += 1
        cluster_end = row.window_end
    else:
        cluster_end = max(
            cluster_end,
            row.window_end,
        )

    cluster_ids.append(cluster_id)


comparison["cluster_id"] = cluster_ids


# Sensitivity sample: retain the first event in each cluster
cluster_first = comparison.drop_duplicates(
    subset=["cluster_id"],
    keep="first",
)


OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_FILE, index=False)


def print_summary(dataframe, sample_name):
    mean_gap_day1 = dataframe["gap_day1"].mean()
    mean_gap_day2 = dataframe["gap_day2"].mean()
    mean_gap_day3 = dataframe["gap_day3"].mean()

    mean_gap_change = dataframe[
        "gap_change_day1_to_day3"
    ].mean()

    median_gap_change = dataframe[
        "gap_change_day1_to_day3"
    ].median()

    narrowing_share = (
        dataframe["gap_change_day1_to_day3"] < 0
    ).mean()

    print(f"\n{sample_name}")
    print(f"Observations: {len(dataframe)}")

    print(
        f"Mean gap Day 1: "
        f"{mean_gap_day1 * 100:.3f} pp"
    )

    print(
        f"Mean gap Day 2: "
        f"{mean_gap_day2 * 100:.3f} pp"
    )

    print(
        f"Mean gap Day 3: "
        f"{mean_gap_day3 * 100:.3f} pp"
    )

    print(
        f"Mean gap change, Day 1 to Day 3: "
        f"{mean_gap_change * 100:.3f} pp"
    )

    print(
        f"Median gap change, Day 1 to Day 3: "
        f"{median_gap_change * 100:.3f} pp"
    )

    print(
        f"Share of events with a narrowing gap: "
        f"{narrowing_share:.1%}"
    )


print(f"Matched events: {len(comparison)}")
print(f"Overlapping-window clusters: {comparison['cluster_id'].nunique()}")


print_summary(
    comparison,
    "Full event sample",
)

print_summary(
    cluster_first,
    "First event from each overlapping cluster",
)


preview = comparison[
    [
        "us_event_date",
        "sp500_return_percent",
        "gap_day1",
        "gap_day3",
        "gap_change_day1_to_day3",
        "cluster_id",
    ]
].head(10).copy()

for column in [
    "gap_day1",
    "gap_day3",
    "gap_change_day1_to_day3",
]:
    preview[column] = preview[column] * 100


print("\nFirst 10 comparisons:")
print(preview.to_string(index=False))

print(f"\nResults saved to: {OUTPUT_FILE.resolve()}")