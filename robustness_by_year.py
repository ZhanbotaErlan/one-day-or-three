from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/market_comparison.csv")

OUTPUT_FOLDER = Path("results")
OUTPUT_FILE = OUTPUT_FOLDER / "year_robustness.csv"


comparison = pd.read_csv(
    INPUT_FILE,
    parse_dates=["us_event_date"],
)

comparison["year"] = comparison[
    "us_event_date"
].dt.year


# Results within each calendar year
annual_summary = (
    comparison
    .groupby("year")
    .agg(
        events=("us_event_date", "count"),
        clusters=("cluster_id", "nunique"),
        mean_gap_day1=("gap_day1", "mean"),
        mean_gap_day3=("gap_day3", "mean"),
        mean_gap_change=(
            "gap_change_day1_to_day3",
            "mean",
        ),
    )
    .reset_index()
)


# Leave one calendar year out at a time
leave_one_year_out = []

for excluded_year in sorted(comparison["year"].unique()):
    sample = comparison[
        comparison["year"] != excluded_year
    ]

    leave_one_year_out.append(
        {
            "excluded_year": excluded_year,
            "events": len(sample),
            "clusters": sample["cluster_id"].nunique(),
            "mean_gap_day1": sample["gap_day1"].mean(),
            "mean_gap_day3": sample["gap_day3"].mean(),
            "mean_gap_change": sample[
                "gap_change_day1_to_day3"
            ].mean(),
        }
    )


leave_one_year_out = pd.DataFrame(
    leave_one_year_out
)


# Special check excluding the COVID crash year
without_2020 = comparison[
    comparison["year"] != 2020
]


print("Events and mean gaps within each year:")

annual_display = annual_summary.copy()

for column in [
    "mean_gap_day1",
    "mean_gap_day3",
    "mean_gap_change",
]:
    annual_display[column] = (
        annual_display[column] * 100
    )

print(annual_display.to_string(index=False))


print("\nResults excluding 2020:")
print(f"Events: {len(without_2020)}")
print(
    f"Clusters: "
    f"{without_2020['cluster_id'].nunique()}"
)
print(
    f"Mean gap Day 1: "
    f"{without_2020['gap_day1'].mean() * 100:.3f} pp"
)
print(
    f"Mean gap Day 3: "
    f"{without_2020['gap_day3'].mean() * 100:.3f} pp"
)
print(
    f"Mean gap change: "
    f"{without_2020['gap_change_day1_to_day3'].mean() * 100:.3f} pp"
)


print("\nLeave-one-year-out results:")

leave_one_year_display = leave_one_year_out.copy()

for column in [
    "mean_gap_day1",
    "mean_gap_day3",
    "mean_gap_change",
]:
    leave_one_year_display[column] = (
        leave_one_year_display[column] * 100
    )

print(
    leave_one_year_display.to_string(
        index=False
    )
)


OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

leave_one_year_out.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"\nResults saved to: "
    f"{OUTPUT_FILE.resolve()}"
)