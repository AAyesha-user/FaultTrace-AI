import pandas as pd


def evaluate_diagnosis(rankings, ground_truth):
    results = []

    for run_id in ground_truth["run_id"].unique():
        run_rankings = rankings[
            rankings["run_id"] == run_id
        ].copy()

        run_rankings = run_rankings.sort_values(
            "suspiciousness", ascending=False
        )

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
            "top1": best_rank == 1 if best_rank is not None else False,
            "top3": best_rank <= 3 if best_rank is not None else False,
            "reciprocal_rank": (
                1 / best_rank if best_rank is not None else 0
            ),
        })

    results_df = pd.DataFrame(results)

    summary = {
        "top1_accuracy": results_df["top1"].mean(),
        "top3_accuracy": results_df["top3"].mean(),
        "mrr": results_df["reciprocal_rank"].mean(),
        "total_runs": len(results_df),
    }

    return results_df, summary