
from pathlib import Path

import pandas as pd

from backend.simulation.engine import WarehouseSimulation
from backend.diagnosis.suspiciousness import calculate_ochiai


def main():
    num_runs = 30
    num_tasks = 100

    faulty_robot_ids = ["R3"]
    fault_probability = 0.30

    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_diagnoses = []
    ground_truth = []

    for run_id in range(1, num_runs + 1):

        simulation = WarehouseSimulation(
            num_robots=5,
            seed=1000 + run_id,
            faulty_robot_ids=faulty_robot_ids,
            fault_probability=fault_probability,
        )

        simulation.generate_tasks(num_tasks=num_tasks)
        records = simulation.run()

        records_dict = [record.to_dict() for record in records]
        run_df = pd.DataFrame(records_dict)
        run_df["run_id"] = run_id

        all_records.extend(run_df.to_dict("records"))

        diagnosis = calculate_ochiai(run_df)
        diagnosis["run_id"] = run_id

        all_diagnoses.append(diagnosis)

        ground_truth.append({
            "run_id": run_id,
            "faulty_robot_ids": ",".join(faulty_robot_ids),
            "fault_probability": fault_probability,
            "num_tasks": num_tasks,
            "seed": 1000 + run_id,
        })

        print(f"Completed experiment {run_id}/{num_runs}")

    history_df = pd.DataFrame(all_records)
    history_df.to_csv(
        output_dir / "execution_history.csv",
        index=False,
    )

    diagnoses_df = pd.concat(
        all_diagnoses,
        ignore_index=True,
    )

    diagnoses_df.to_csv(
        output_dir / "diagnostic_rankings.csv",
        index=False,
    )

    ground_truth_df = pd.DataFrame(ground_truth)
    ground_truth_df.to_csv(
        output_dir / "ground_truth.csv",
        index=False,
    )

    top_rank_matches = 0

    for diagnosis in all_diagnoses:
        top_robot = diagnosis.iloc[0]["robot_id"]

        if top_robot in faulty_robot_ids:
            top_rank_matches += 1

    top1_accuracy = top_rank_matches / num_runs

    print("\n=== FaultTrace AI Experiment Report ===")
    print(f"Experiment runs: {num_runs}")
    print(f"Tasks per run: {num_tasks}")
    print(f"Known faulty robots: {faulty_robot_ids}")
    print(f"Fault activation probability: {fault_probability}")
    print(f"Top-1 identification rate: {top1_accuracy:.2%}")

    print("\nFiles saved:")
    print(output_dir / "execution_history.csv")
    print(output_dir / "diagnostic_rankings.csv")
    print(output_dir / "ground_truth.csv")


if __name__ == "__main__":
    main()
