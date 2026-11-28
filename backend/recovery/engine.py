"""
FaultTrace AI - Recovery engine (honest version)
------------------------------------------------
Save as:  backend/recovery/engine.py
(BACK UP your old file first, for example copy it to engine_old.py)
 
What was wrong before:
    every reassigned task was simply marked  recovery_success = True.
    That makes "100% recovery" meaningless.
 
What this version does:
    1. Finds the failed tasks of the isolated (suspect) robots.
    2. Picks a replacement robot for each task from the robots that were NOT
       isolated (the least busy one first; ties go to the more reliable one).
    3. RE-ATTEMPTS the task on that robot. The attempt succeeds or fails
       depending on how reliable that robot really was in the execution
       history. So if you isolate the wrong robot and a faulty robot takes
       over, some retries fail, and the recovery rate drops below 100%.
    4. Records the real outcome in  recovery_success.
 
The function name and the return value are the same as before, so app.py
and api.py do not need any change:
    recovered_df, summary = simulate_recovery(history_df, ["R3"])
"""
 
import random
 
import pandas as pd
 
 
def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
 
 
def _attempt_task(target_robot, reliability, rng):
    """
    One re-execution of a task on `target_robot`.
 
    Returns (success: bool, success_probability: float).
 
    If your WarehouseSimulation has a method that executes ONE task on ONE
    robot, call it here instead of the random draw. Everything else in this
    file can stay the same.
    """
    probability = float(reliability[target_robot])
    return rng.random() < probability, probability
 
 
def simulate_recovery(history: pd.DataFrame, suspects, seed: int = 42):
    if history is None or history.empty:
        raise ValueError("The execution history is empty.")
 
    for column in ("robot_id", "success"):
        if column not in history.columns:
            raise ValueError(f"The execution history has no '{column}' column.")
 
    df = history.copy()
    df["success"] = _as_bool(df["success"])
    df = df.dropna(subset=["robot_id", "success"])
 
    suspects = list(dict.fromkeys(suspects))  # remove duplicates, keep order
    robots = sorted(df["robot_id"].unique().tolist())
    eligible = [r for r in robots if r not in suspects]
 
    if not eligible:
        raise ValueError(
            "Every robot is isolated, so no healthy robot is left to take over the tasks."
        )
 
    # How reliable each robot really was in the observed history.
    reliability = df.groupby("robot_id")["success"].mean()
 
    failed = df[df["robot_id"].isin(suspects) & (~df["success"])]
 
    rng = random.Random(seed)
    load = {robot: 0 for robot in eligible}
    rows = []
 
    for _, task in failed.iterrows():
        # Least busy replacement first; ties go to the more reliable robot.
        target = min(eligible, key=lambda r: (load[r], -float(reliability[r])))
        load[target] += 1
 
        success, probability = _attempt_task(target, reliability, rng)
 
        record = task.to_dict()
        record["original_robot"] = task["robot_id"]
        record["reassigned_to"] = target
        record["retry_success_probability"] = round(probability, 3)
        record["recovery_success"] = bool(success)
        rows.append(record)
 
    recovered = pd.DataFrame(rows)
 
    reassigned = int(len(recovered))
    succeeded = int(recovered["recovery_success"].sum()) if reassigned else 0
 
    summary = {
        "failed_tasks_identified": reassigned,
        "tasks_reassigned": reassigned,
        "successful_recoveries": succeeded,
        "tasks_still_failed": reassigned - succeeded,
        "recovery_rate": round(100 * succeeded / reassigned, 1) if reassigned else 0.0,
        "isolated_robots": suspects,
        "replacement_robots": eligible,
        "method": "Each failed task is re-attempted on a replacement robot; "
                  "success depends on that robot's observed reliability.",
    }
 
    return recovered, summary
 