
import pandas as pd


def simulate_recovery(history, suspicious_robot_ids):
    """
    Simulate recovery by reallocating failed tasks from
    suspicious robots to healthier robots.
    """

    if history.empty:
        raise ValueError("Execution history is empty.")

    required = {"robot_id", "success", "task_id"}
    missing = required - set(history.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = history.copy()

    if df["success"].dtype != bool:
        df["success"] = (
            df["success"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    suspicious_robot_ids = set(suspicious_robot_ids)

    robot_stats = (
        df.groupby("robot_id")
        .agg(
            total_tasks=("success", "count"),
            successful_tasks=("success", "sum")
        )
    )

    robot_stats["success_rate"] = (
        robot_stats["successful_tasks"]
        / robot_stats["total_tasks"]
    )

    healthy = robot_stats.drop(
        index=[
            robot for robot in suspicious_robot_ids
            if robot in robot_stats.index
        ],
        errors="ignore"
    )

    if healthy.empty:
        raise ValueError(
            "No healthy robots available for recovery."
        )

    healthy = healthy.sort_values(
        ["success_rate", "successful_tasks"],
        ascending=False
    )

    failed_tasks = df[
        (~df["success"])
        & (df["robot_id"].isin(suspicious_robot_ids))
    ].copy()

    recovered = []

    for _, task in failed_tasks.iterrows():

        replacement = healthy.index[
            len(recovered) % len(healthy)
        ]

        record = task.to_dict()

        record["original_robot"] = task["robot_id"]
        record["reassigned_robot"] = replacement
        record["recovery_success"] = True
        record["recovery_status"] = "Recovered"

        recovered.append(record)

    recovery_df = pd.DataFrame(recovered)

    total_failed = len(failed_tasks)
    total_recovered = len(recovery_df)

    summary = {
        "suspicious_robots": list(suspicious_robot_ids),
        "failed_tasks_identified": total_failed,
        "tasks_reassigned": total_recovered,
        "successful_recoveries": total_recovered,
        "recovery_rate": (
            total_recovered / total_failed * 100
            if total_failed else 0
        ),
        "healthy_robots_available": len(healthy)
    }

    return recovery_df, summary
