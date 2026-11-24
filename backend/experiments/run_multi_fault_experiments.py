from pathlib import Path

import pandas as pd

from backend.simulation.engine import WarehouseSimulation
from backend.diagnosis.suspiciousness import calculate_ochiai


def evaluate_multi_fault(diagnosis, faulty_robot_ids):
    ranked_robots = diagnosis.sort_values(
        "suspiciousness", ascending=False
    )["robot_id"].tolist()

    ranks = {
        robot_id: rank
        for rank, robot_id in enumerate(ranked_robots, start=1)
    }

    faulty_ranks = [
        ranks[robot_id]
        for robot_id in faulty_robot_ids
        if robot_id in ranks
    ]

    top1_hit = any(rank == 1 for rank in faulty_ranks)
    top3_hit = any(rank <= 3 for rank in faulty_ranks)

    recall_at_3 = sum(
        rank <= 3 for rank in faulty_ranks
    ) / len(faulty_robot_ids)

    mean_fault_rank = (
        sum(faulty_ranks) / len(faulty_ranks)
        if faulty_ranks else 0
    )

    mean_reciprocal_rank = (
        sum(1 / rank for rank in faulty_ranks)
        / len(faulty_ranks)
        if faulty_ranks else 0
    )

    return {
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "fault_recall_at_3": recall_at_3,
        "mean_fault_rank": mean_fault_rank,
        "mrr_faults": mean_reciprocal_rank,
    }


def main():
    num_runs = 30
    num_tasks = 100
    fault_probability = 0.30

    scenarios = {
        "M2": ["R2", "R3"],
        "M3": ["R1", "R3", "R5"],
    }

    output_dir = Path("data/multi_fault")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_rankings = []
    all_evaluations = []

    for scenario, faulty_robot_ids in scenarios.items():

        print(f"\nStarting {scenario}: {faulty_robot_ids}")

        for run_id in range(1, num_runs + 1):

            simulation = WarehouseSimulation(
                num_robots=5,
                seed=5000 + run_id,
                faulty_robot_ids=faulty_robot_ids,
                fault_probability=fault_probability,
            )

            simulation.generate_tasks(num_tasks=num_tasks)
            records = simulation.run()

            records_dict = [
                record.to_dict() for record in records
            ]

            run_df = pd.DataFrame(records_dict)
            run_df["run_id"] = run_id
            run_df["scenario"] = scenario

            all_records.extend(records_dict)

            diagnosis = calculate_ochiai(run_df)

            diagnosis["run_id"] = run_id
            diagnosis["scenario"] = scenario

            all_rankings.append(diagnosis)

            evaluation = evaluate_multi_fault(
                diagnosis, faulty_robot_ids
            )

            evaluation.update({
                "run_id": run_id,
                "scenario": scenario,
                "faulty_robot_ids": ",".join(
                    faulty_robot_ids
                ),
            })

            all_evaluations.append(evaluation)

            print(
                f"{scenario}: run {run_id}/{num_runs} completed"
            )

        print(f"{scenario} completed.")

    history_df = pd.DataFrame(all_records)
    rankings_df = pd.concat(
        all_rankings, ignore_index=True
    )
    evaluations_df = pd.DataFrame(all_evaluations)

    history_df.to_csv(
        output_dir / "execution_history.csv",
        index=False,
    )

    rankings_df.to_csv(
        output_dir / "diagnostic_rankings.csv",
        index=False,
    )

    evaluations_df.to_csv(
        output_dir / "evaluation_per_run.csv",
        index=False,
    )

    summary = evaluations_df.groupby("scenario").agg(
        top1_hit_rate=("top1_hit", "mean"),
        top3_hit_rate=("top3_hit", "mean"),
        fault_recall_at_3=("fault_recall_at_3", "mean"),
        mean_fault_rank=("mean_fault_rank", "mean"),
        mrr_faults=("mrr_faults", "mean"),
    ).reset_index()

    summary.to_csv(
        output_dir / "scenario_summary.csv",
        index=False,
    )

    print("\n=== FaultTrace AI: Multi-Fault Report ===")
    print(summary.to_string(index=False))

    print("\nFiles saved:")
    print(output_dir / "execution_history.csv")
    print(output_dir / "diagnostic_rankings.csv")
    print(output_dir / "evaluation_per_run.csv")
    print(output_dir / "scenario_summary.csv")


if __name__ == "__main__":
    main()