from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

REPETITIONS = 10_000
SEED = 42

METRICS = [
    "gap_day1",
    "gap_day3",
    "gap_change_day1_to_day3",
]

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# Read the final calendar-filtered sample and full daily panel.
strict = pd.read_csv(
    PROCESSED / "market_comparison_strict.csv",
    parse_dates=["us_event_date", "window_start", "window_end"],
)

panel = pd.read_csv(
    PROCESSED / "strict_benchmark_panel.csv",
    parse_dates=["date", "window_start", "window_end"],
)

benchmark = pd.read_csv(
    RESULTS / "strict_calm_benchmark.csv"
)

strict["year"] = strict["us_event_date"].dt.year

panel["gap_change_day1_to_day3"] = (
    panel["gap_day3"] - panel["gap_day1"]
)


# Rebuild overlap groups whenever the event sample changes.
def assign_clusters(dataframe):
    dataframe = dataframe.sort_values(
        ["window_start", "window_end"]
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
    return dataframe


# Calculate event-weighted means and cluster-bootstrap intervals.
def summarize_sample(dataframe):
    if dataframe.empty or dataframe[METRICS].isna().any().any():
        raise ValueError("Empty sample or missing outcome values.")

    dataframe = assign_clusters(dataframe)
    groups = dataframe.groupby("cluster_id")

    sums = groups[METRICS].sum().to_numpy()
    counts = groups.size().to_numpy()

    if len(counts) < 2:
        raise ValueError("Too few overlap groups.")

    rng = np.random.default_rng(SEED)

    sampled = rng.integers(
        0,
        len(counts),
        size=(REPETITIONS, len(counts)),
    )

    bootstrap = (
        sums[sampled].sum(axis=1)
        / counts[sampled].sum(axis=1)[:, None]
    )

    lower, upper = np.quantile(
        bootstrap, [0.025, 0.975], axis=0
    )

    return pd.DataFrame(
        {
            "metric": METRICS,
            "events": len(dataframe),
            "clusters": len(counts),
            "estimate_pp": dataframe[METRICS].mean().to_numpy() * 100,
            "ci_lower_pp": lower * 100,
            "ci_upper_pp": upper * 100,
        }
    )


# Main results
main_summary = summarize_sample(strict)

main_summary.to_csv(
    RESULTS / "final_main_results.csv",
    index=False,
)


# Repeat all thresholds with the same calendar filter.
threshold_tables = []

for threshold in [-0.015, -0.020, -0.025]:
    sample = panel[
        panel["eligible"]
        & (panel["sp500_return"] <= threshold)
    ].copy()

    if threshold == -0.020:
        if set(sample["date"]) != set(strict["us_event_date"]):
            raise ValueError("The -2% event sample does not match.")

        actual = sample.set_index("date")
        expected = strict.set_index("us_event_date")

        if not np.allclose(
            actual.loc[expected.index, METRICS].to_numpy(),
            expected[METRICS].to_numpy(),
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("The -2% return values do not match.")

    table = summarize_sample(sample)
    table.insert(0, "threshold_percent", threshold * 100)
    threshold_tables.append(table)

threshold_summary = pd.concat(
    threshold_tables, ignore_index=True
)

threshold_summary.to_csv(
    RESULTS / "final_threshold_results.csv",
    index=False,
)


# Repeat leave-one-year-out checks on the filtered sample.
year_tables = []

for year in sorted(strict["year"].unique()):
    sample = strict[strict["year"] != year].copy()

    table = summarize_sample(sample)
    table.insert(0, "excluded_year", year)
    year_tables.append(table)

year_summary = pd.concat(year_tables, ignore_index=True)

year_summary.to_csv(
    RESULTS / "final_year_results.csv",
    index=False,
)


# Plot style
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)

NAVY = "#24476B"
TEAL = "#168A83"


def save_figure(figure, filename):
    for extension in ["png", "pdf"]:
        figure.savefig(
            FIGURES / f"{filename}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(figure)


# Figure 1 — cumulative mean returns
figure, axis = plt.subplots(figsize=(8, 5))

path_rows = []

for market, label, color in [
    ("hsi", "Hang Seng Index", NAVY),
    ("kase", "KASE Index", TEAL),
]:
    means = [0.0]

    for day in [1, 2, 3]:
        value = strict[f"{market}_return_day{day}"].mean() * 100
        means.append(value)

        path_rows.append(
            {
                "market": label,
                "local_session": day,
                "mean_return_percent": value,
                "events": len(strict),
            }
        )

    axis.plot(
        [0, 1, 2, 3],
        means,
        marker="o",
        linewidth=2,
        color=color,
        label=label,
    )

axis.axhline(0, color="gray", linewidth=0.8)
axis.set_xticks([0, 1, 2, 3])
axis.set_xticklabels(["Baseline", "Day 1", "Day 2", "Day 3"])
axis.set_ylabel("Mean cumulative price-index return (%)")
axis.set_title("Returns following U.S. sell-offs")
axis.legend(frameon=False)

figure.text(
    0.1, 0.02,
    f"{len(strict)} calendar-filtered events; S&P 500 return <= -2%.\n"
    "Local-currency returns; each market follows its own trading calendar.\n"
    "Descriptive means; uncertainty in the paired gaps is shown in Figure 2.",
    fontsize=9,
)

figure.tight_layout(rect=[0, 0.14, 1, 1])
save_figure(figure, "figure_1_return_paths")

pd.DataFrame(path_rows).to_csv(
    RESULTS / "final_return_paths.csv",
    index=False,
)


# Helper for estimates and horizontal confidence intervals.
def plot_intervals(axis, table, labels, color):
    positions = np.arange(len(table))

    axis.hlines(
        positions,
        table["ci_lower_pp"],
        table["ci_upper_pp"],
        color=color,
        linewidth=2,
    )

    axis.scatter(
        table["estimate_pp"],
        positions,
        color=color,
        s=45,
        zorder=3,
    )

    axis.axvline(0, color="gray", linestyle="--", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_ylim(len(table) - 0.5, -0.5)
    axis.set_xlabel("Percentage points")


# Figure 2 — show the original research question alongside Day 1.
labels = ["Day 1 gap", "Day 3 gap", "Change, Day 1 to Day 3"]

adjusted_order = [
    "Day 1 excess",
    "Day 3 excess",
    "Change in excess, Day 1 to Day 3",
]

adjusted = (
    benchmark[benchmark["block_length"] == 20]
    .set_index("metric")
    .loc[adjusted_order]
    .reset_index()
)

figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

plot_intervals(axes[0], main_summary, labels, NAVY)
axes[0].set_title("KASE minus Hang Seng")

plot_intervals(axes[1], adjusted, labels, TEAL)
axes[1].set_title("After subtracting the calm-day benchmark")

# Use the same horizontal scale for both panels.
lower_limit = min(
    main_summary["ci_lower_pp"].min(),
    adjusted["ci_lower_pp"].min(),
) - 0.15

upper_limit = max(
    main_summary["ci_upper_pp"].max(),
    adjusted["ci_upper_pp"].max(),
) + 0.15

for axis in axes:
    axis.set_xlim(lower_limit, upper_limit)

figure.text(
    0.06, 0.02,
    "Separate 95% intervals, without a multiple-testing adjustment.\n"
    "Left: overlap-cluster bootstrap. Right: within-year circular block bootstrap, "
    "20 U.S. sessions per block.\n"
    "The change from Day 1 to Day 3 is the originally specified primary comparison.",
    fontsize=9,
)

figure.tight_layout(rect=[0, 0.16, 1, 1])
save_figure(figure, "figure_2_gaps_and_intervals")


# Figure 3 — threshold sensitivity after calendar filtering.
day1_thresholds = threshold_summary[
    threshold_summary["metric"] == "gap_day1"
].reset_index(drop=True)

threshold_labels = [
    f"At most {row.threshold_percent:.1f}%  "
    f"(n={int(row.events)})"
    for row in day1_thresholds.itertuples()
]

figure, axis = plt.subplots(figsize=(8, 4.5))

plot_intervals(
    axis, day1_thresholds, threshold_labels, NAVY
)

axis.set_title("Day 1 gap across U.S. sell-off thresholds")
axis.set_ylabel("S&P 500 daily return")

figure.text(
    0.1, 0.02,
    "Same calendar filter at every threshold; 95% overlap-cluster bootstrap intervals.\n"
    "Threshold samples overlap and are not independent replications.",
    fontsize=9,
)

figure.tight_layout(rect=[0, 0.14, 1, 1])
save_figure(figure, "figure_3_threshold_sensitivity")


print("\nFinal calendar-filtered threshold results:")
print(threshold_summary.round(3).to_string(index=False))

print("\nLeave-one-year-out Day 1 results:")
print(
    year_summary[year_summary["metric"] == "gap_day1"]
    .round(3)
    .to_string(index=False)
)

print(f"\nTables saved in: {RESULTS}")
print(f"Three figures saved as PNG and PDF in: {FIGURES}")