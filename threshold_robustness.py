from pathlib import Path

import numpy as np
import pandas as pd


SP500_FILE = Path("data/raw/sp500.csv")
HSI_FILE = Path("data/raw/hsi.csv")
KASE_FILE = Path("data/processed/kase_clean.csv")

OUTPUT_FOLDER = Path("results")
OUTPUT_FILE = OUTPUT_FOLDER / "threshold_robustness.csv"

THRESHOLDS = [-0.015, -0.020, -0.025]

BOOTSTRAP_REPETITIONS = 10_000
RANDOM_SEED = 42


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


def build_comparison(events):
    rows = []

    for event in events.itertuples():
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

        row = {
            "us_event_date": event.date,
            "sp500_return": event.daily_return,
            "hsi_baseline_date": hsi_baseline["date"],
            "kase_baseline_date": kase_baseline["date"],
        }

        for day in [1, 2, 3]:
            hsi_day = hsi_future.iloc[day - 1]
            kase_day = kase_future.iloc[day - 1]

            row[f"hsi_day{day}_date"] = hsi_day["date"]
            row[f"kase_day{day}_date"] = kase_day["date"]

            row[f"hsi_return_day{day}"] = (
                hsi_day["close"]
                / hsi_baseline["close"]
                - 1
            )

            row[f"kase_return_day{day}"] = (
                kase_day["close"]
                / kase_baseline["close"]
                - 1
            )

        rows.append(row)

    comparison = pd.DataFrame(rows)

    for day in [1, 2, 3]:
        comparison[f"gap_day{day}"] = (
            comparison[f"kase_return_day{day}"]
            - comparison[f"hsi_return_day{day}"]
        )

    comparison["gap_change"] = (
        comparison["gap_day3"]
        - comparison["gap_day1"]
    )

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

    cluster_ids = []
    cluster_id = 0
    cluster_end = None

    for row in comparison.itertuples():
        if (
            cluster_end is None
            or row.window_start > cluster_end
        ):
            cluster_id += 1
            cluster_end = row.window_end
        else:
            cluster_end = max(
                cluster_end,
                row.window_end,
            )

        cluster_ids.append(cluster_id)

    comparison["cluster_id"] = cluster_ids

    return comparison


def cluster_bootstrap_interval(
    comparison,
    column,
    random_seed,
):
    cluster_statistics = (
        comparison
        .groupby("cluster_id")[column]
        .agg(["sum", "count"])
    )

    cluster_sums = cluster_statistics["sum"].to_numpy()
    cluster_counts = cluster_statistics["count"].to_numpy()

    number_of_clusters = len(cluster_statistics)

    rng = np.random.default_rng(random_seed)

    estimates = np.empty(
        BOOTSTRAP_REPETITIONS
    )

    for repetition in range(BOOTSTRAP_REPETITIONS):
        sampled_indices = rng.integers(
            0,
            number_of_clusters,
            size=number_of_clusters,
        )

        estimates[repetition] = (
            cluster_sums[sampled_indices].sum()
            / cluster_counts[sampled_indices].sum()
        )

    lower = np.quantile(estimates, 0.025)
    upper = np.quantile(estimates, 0.975)

    return lower, upper


summary_rows = []


for threshold_number, threshold in enumerate(THRESHOLDS):
    events = sp500[
        sp500["daily_return"] <= threshold
    ][
        ["date", "daily_return"]
    ].copy()

    comparison = build_comparison(events)

    day1_lower, day1_upper = cluster_bootstrap_interval(
        comparison,
        "gap_day1",
        RANDOM_SEED + threshold_number,
    )

    day3_lower, day3_upper = cluster_bootstrap_interval(
        comparison,
        "gap_day3",
        RANDOM_SEED + threshold_number,
    )

    change_lower, change_upper = cluster_bootstrap_interval(
        comparison,
        "gap_change",
        RANDOM_SEED + threshold_number,
    )

    summary_rows.append(
        {
            "threshold_percent": threshold * 100,
            "raw_events": len(events),
            "matched_events": len(comparison),
            "clusters": comparison["cluster_id"].nunique(),

            "mean_gap_day1_pp":
                comparison["gap_day1"].mean() * 100,
            "day1_ci_lower_pp": day1_lower * 100,
            "day1_ci_upper_pp": day1_upper * 100,

            "mean_gap_day3_pp":
                comparison["gap_day3"].mean() * 100,
            "day3_ci_lower_pp": day3_lower * 100,
            "day3_ci_upper_pp": day3_upper * 100,

            "mean_gap_change_pp":
                comparison["gap_change"].mean() * 100,
            "change_ci_lower_pp": change_lower * 100,
            "change_ci_upper_pp": change_upper * 100,
        }
    )

    print(
        f"\nSell-off threshold: "
        f"{threshold * 100:.1f}%"
    )

    print(f"Raw events: {len(events)}")
    print(f"Matched events: {len(comparison)}")
    print(
        f"Overlapping clusters: "
        f"{comparison['cluster_id'].nunique()}"
    )

    print(
        f"Day 1 gap: "
        f"{comparison['gap_day1'].mean() * 100:.3f} pp "
        f"[{day1_lower * 100:.3f}, "
        f"{day1_upper * 100:.3f}]"
    )

    print(
        f"Day 3 gap: "
        f"{comparison['gap_day3'].mean() * 100:.3f} pp "
        f"[{day3_lower * 100:.3f}, "
        f"{day3_upper * 100:.3f}]"
    )

    print(
        f"Gap change: "
        f"{comparison['gap_change'].mean() * 100:.3f} pp "
        f"[{change_lower * 100:.3f}, "
        f"{change_upper * 100:.3f}]"
    )


summary = pd.DataFrame(summary_rows)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT_FILE, index=False)

print(
    f"\nResults saved to: "
    f"{OUTPUT_FILE.resolve()}"
)