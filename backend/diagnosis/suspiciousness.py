import pandas as pd
from pathlib import Path


def calculate_ochiai(history: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Ochiai suspiciousness for each robot.

    Robots with a stronger association with failed tasks
    receive higher suspiciousness scores.
    """

    total_failed = int((history["success"] == False).sum())

    results = []

    for robot_id, group in history.groupby("robot_id"):

        failed_by_robot = int((group["success"] == False).sum())
        tasks_by_robot = len(group)

        if total_failed == 0 or tasks_by_robot == 0:
            score = 0.0
        else:
            score = failed_by_robot / (
                (total_failed * tasks_by_robot) ** 0.5
            )

        results.append({
            "robot_id": robot_id,
            "total_tasks": tasks_by_robot,
            "failed_tasks": failed_by_robot,
            "suspiciousness": score,
        })

    result = pd.DataFrame(results)

    result = result.sort_values(
        by="suspiciousness",
        ascending=False
    ).reset_index(drop=True)

    return result


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    data_path = project_root / "data" / "execution_history.csv"

    history = pd.read_csv(data_path)

    diagnosis = calculate_ochiai(history)

    print("\n=== FaultTrace AI: Robot Diagnosis ===\n")
    print(diagnosis.to_string(index=False))