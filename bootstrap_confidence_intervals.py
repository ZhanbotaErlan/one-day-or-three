from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/market_comparison.csv")

OUTPUT_FOLDER = Path("results")
OUTPUT_FILE = OUTPUT_FOLDER / "cluster_bootstrap.csv"

BOOTSTRAP_REPETITIONS = 10_000
RANDOM_SEED = 42


comparison = pd.read_csv(INPUT_FILE)


# Store every overlapping event cluster separately
clusters = {
    cluster_id: cluster.copy()
    for cluster_id, cluster in comparison.groupby("cluster_id")
}

cluster_ids = np.array(list(clusters.keys()))

print(f"Events: {len(comparison)}")
print(f"Independent event clusters: {len(cluster_ids)}")
print(f"Bootstrap repetitions: {BOOTSTRAP_REPETITIONS}")


rng = np.random.default_rng(RANDOM_SEED)

bootstrap_results = []


for repetition in range(BOOTSTRAP_REPETITIONS):
    # Sample complete clusters with replacement
    sampled_cluster_ids = rng.choice(
        cluster_ids,
        size=len(cluster_ids),
        replace=True,
    )

    sampled_data = pd.concat(
        [
            clusters[cluster_id]
            for cluster_id in sampled_cluster_ids
        ],
        ignore_index=True,
    )

    bootstrap_results.append(
        {
            "mean_gap_day1": sampled_data["gap_day1"].mean(),
            "mean_gap_day2": sampled_data["gap_day2"].mean(),
            "mean_gap_day3": sampled_data["gap_day3"].mean(),
            "mean_gap_change": sampled_data[
                "gap_change_day1_to_day3"
            ].mean(),
        }
    )


bootstrap = pd.DataFrame(bootstrap_results)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
bootstrap.to_csv(OUTPUT_FILE, index=False)


def print_confidence_interval(
    original_column,
    bootstrap_column,
    label,
):
    estimate = comparison[original_column].mean()

    lower_bound = bootstrap[
        bootstrap_column
    ].quantile(0.025)

    upper_bound = bootstrap[
        bootstrap_column
    ].quantile(0.975)

    print(f"\n{label}")
    print(f"Estimate: {estimate * 100:.3f} pp")
    print(
        f"95% cluster-bootstrap CI: "
        f"[{lower_bound * 100:.3f}, "
        f"{upper_bound * 100:.3f}] pp"
    )

    if lower_bound > 0:
        print("Interval is entirely above zero.")
    elif upper_bound < 0:
        print("Interval is entirely below zero.")
    else:
        print("Interval includes zero.")


print_confidence_interval(
    "gap_day1",
    "mean_gap_day1",
    "Mean KASE–HSI gap on Day 1",
)

print_confidence_interval(
    "gap_day2",
    "mean_gap_day2",
    "Mean KASE–HSI gap on Day 2",
)

print_confidence_interval(
    "gap_day3",
    "mean_gap_day3",
    "Mean KASE–HSI gap on Day 3",
)

print_confidence_interval(
    "gap_change_day1_to_day3",
    "mean_gap_change",
    "Change in the gap from Day 1 to Day 3",
)


probability_narrowing = (
    bootstrap["mean_gap_change"] < 0
).mean()

print(
    "\nShare of bootstrap estimates "
    f"with a narrowing mean gap: "
    f"{probability_narrowing:.1%}"
)

print(
    f"\nBootstrap distribution saved to: "
    f"{OUTPUT_FILE.resolve()}"
)