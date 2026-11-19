import pandas as pd
from pathlib import Path


def evaluate_method(rankings, ground_truth, method_name):
    results = []

    for run_id in ground_truth["run_id"].unique():
        run_rankings = rankings[
            rankings["run_id"] == run_id
        ].sort_values("score", ascending=False)

        truth_row = ground_truth[
            ground_truth["run_id"] == run_id
        ].iloc[0]

        faulty_robots = set(
            str(truth_row["faulty_robot_ids"]).split(",")
        )

        ranked_robots = run_rankings["robot_id"].tolist()

        ranks = [
            ranked_robots.index(robot) + 1
            for robot in faulty_robots
            if robot in ranked_robots
        ]

        best_rank = min(ranks) if ranks else None

        results.append({
            "run_id": run_id,
            "method": method_name,
            "top1": best_rank == 1 if best_rank is not None else False,
            "top3": best_rank <= 3 if best_rank is not None else False,
            "reciprocal_rank": (
                1 / best_rank if best_rank is not None else 0
            ),
        })

    return pd.DataFrame(results)


def main():
    data_dir = Path("data/experiments")

    history = pd.read_csv(data_dir / "execution_history.csv")
    ground_truth = pd.read_csv(data_dir / "ground_truth.csv")

    # Baseline: rank robots by their task failure rate.
    failure_rankings = (
        history.groupby(["run_id", "robot_id"])
        .agg(
            total_tasks=("success", "count"),
            failed_tasks=("success", lambda x: (~x).sum()),
        )
        .reset_index()
    )

    failure_rankings["score"] = (
        failure_rankings["failed_tasks"]
        / failure_rankings["total_tasks"]
    )

    failure_results = evaluate_method(
        failure_rankings, ground_truth, "Failure Rate"
    )

    # Existing method: Ochiai.
    ochiai = pd.read_csv(
        data_dir / "diagnostic_rankings.csv"
    )

    ochiai_rankings = ochiai.rename(
        columns={"suspiciousness": "score"}
    )

    ochiai_results = evaluate_method(
        ochiai_rankings, ground_truth, "Ochiai"
    )

    combined = pd.concat(
        [failure_results, ochiai_results],
        ignore_index=True,
    )

    combined.to_csv(
        data_dir / "method_comparison_per_run.csv",
        index=False,
    )

    summary = (
        combined.groupby("method")
        .agg(
            top1_accuracy=("top1", "mean"),
            top3_accuracy=("top3", "mean"),
            mrr=("reciprocal_rank", "mean"),
        )
    )

    print("\n=== FaultTrace AI: Method Comparison ===")
    print(summary.to_string())

    print("\nSaved:")
    print(data_dir / "method_comparison_per_run.csv")


if __name__ == "__main__":
    main()