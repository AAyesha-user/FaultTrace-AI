from pathlib import Path

import pandas as pd

from backend.simulation.engine import WarehouseSimulation
from backend.diagnosis.suspiciousness import calculate_ochiai


def main():
    num_runs = 30
    num_tasks = 100

    # Five stress-test scenarios.
    scenarios = [
        {
            "scenario": "S1",
            "faulty_robot_ids": ["R3"],
            "fault_probability": 0.10,
        },
        {
            "scenario": "S2",
            "faulty_robot_ids": ["R3"],
            "fault_probability": 0.30,
        },
        {
            "scenario": "S3",
            "faulty_robot_ids": ["R3"],
            "fault_probability": 0.50,
        },
        {
            "scenario": "S4",
            "faulty_robot_ids": ["R2", "R3"],
            "fault_probability": 0.30,
        },
        {
            "scenario": "S5",
            "faulty_robot_ids": ["R2", "R3", "R5"],
            "fault_probability": 0.30,
        },
    ]

    output_dir = Path("data/stress_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_rankings = []
    all_results = []

    for scenario in scenarios:

        scenario_name = scenario["scenario"]
        faulty_robot_ids = scenario["faulty_robot_ids"]
        fault_probability = scenario["fault_probability"]

        top1_matches = 0
        top3_matches = 0
        reciprocal_ranks = []

        print(f"\nStarting {scenario_name}")

        for run_id in range(1, num_runs + 1):

            seed = 2000 + run_id

            simulation = WarehouseSimulation(
                num_robots=5,
                seed=seed,
                faulty_robot_ids=faulty_robot_ids,
                fault_probability=fault_probability,
            )

            simulation.generate_tasks(num_tasks=num_tasks)
            records = simulation.run()

            records_dict = [record.to_dict() for record in records]

            run_df = pd.DataFrame(records_dict)
            run_df["run_id"] = run_id
            run_df["scenario"] = scenario_name

            all_records.extend(records_dict)

            diagnosis = calculate_ochiai(run_df)

            diagnosis["run_id"] = run_id
            diagnosis["scenario"] = scenario_name

            all_rankings.append(diagnosis)

            ranked_robots = diagnosis["robot_id"].tolist()

            # Rank of the first known faulty robot.
            fault_ranks = [
                ranked_robots.index(robot_id) + 1
                for robot_id in faulty_robot_ids
                if robot_id in ranked_robots
            ]

            best_fault_rank = (
                min(fault_ranks) if fault_ranks else len(ranked_robots) + 1
            )

            top1 = best_fault_rank == 1
            top3 = best_fault_rank <= 3

            reciprocal_rank = 1 / best_fault_rank

            if top1:
                top1_matches += 1

            if top3:
                top3_matches += 1

            reciprocal_ranks.append(reciprocal_rank)

            all_results.append({
                "scenario": scenario_name,
                "run_id": run_id,
                "faulty_robot_ids": ",".join(faulty_robot_ids),
                "fault_probability": fault_probability,
                "top1": top1,
                "top3": top3,
                "best_fault_rank": best_fault_rank,
                "reciprocal_rank": reciprocal_rank,
            })

            print(
                f"{scenario_name}: "
                f"run {run_id}/{num_runs} completed"
            )

        # Scenario-level summary.
        summary = {
            "scenario": scenario_name,
            "faulty_robot_ids": ",".join(faulty_robot_ids),
            "fault_probability": fault_probability,
            "runs": num_runs,
            "tasks_per_run": num_tasks,
            "top1_accuracy": top1_matches / num_runs,
            "top3_accuracy": top3_matches / num_runs,
            "mrr": sum(reciprocal_ranks) / num_runs,
        }

        print(f"\n{scenario_name} summary:")
        print(f"Top-1 accuracy: {summary['top1_accuracy']:.2%}")
        print(f"Top-3 accuracy: {summary['top3_accuracy']:.2%}")
        print(f"MRR: {summary['mrr']:.4f}")

    # Save detailed execution history.
    history_df = pd.DataFrame(all_records)
    history_df.to_csv(
        output_dir / "execution_history.csv",
        index=False,
    )

    # Save all diagnostic rankings.
    rankings_df = pd.concat(
        all_rankings,
        ignore_index=True,
    )

    rankings_df.to_csv(
        output_dir / "diagnostic_rankings.csv",
        index=False,
    )

    # Save individual run metrics.
    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        output_dir / "evaluation_per_run.csv",
        index=False,
    )

    # Calculate final summary across scenarios.
    summary_df = (
        results_df.groupby("scenario")
        .agg(
            top1_accuracy=("top1", "mean"),
            top3_accuracy=("top3", "mean"),
            mrr=("reciprocal_rank", "mean"),
        )
        .reset_index()
    )

    summary_df.to_csv(
        output_dir / "scenario_summary.csv",
        index=False,
    )

    print("\n=== FaultTrace AI: Stress-Test Summary ===")
    print(summary_df.to_string(index=False))

    print("\nFiles saved:")
    print(output_dir / "execution_history.csv")
    print(output_dir / "diagnostic_rankings.csv")
    print(output_dir / "evaluation_per_run.csv")
    print(output_dir / "scenario_summary.csv")


if __name__ == "__main__":
    main()