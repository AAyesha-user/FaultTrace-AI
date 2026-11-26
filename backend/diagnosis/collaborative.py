
import pandas as pd

from backend.diagnosis.suspiciousness import calculate_ochiai


# ==================================================
# AGENT 1: STATISTICAL INVESTIGATOR
# ==================================================

def statistical_agent(history):
    ranking = calculate_ochiai(history)

    return ranking[
        ["robot_id", "suspiciousness"]
    ].rename(
        columns={"suspiciousness": "statistical_score"}
    )


# ==================================================
# AGENT 2: RELIABILITY INVESTIGATOR
# ==================================================

def reliability_agent(history):
    results = []

    for robot_id, group in history.groupby("robot_id"):

        total = len(group)
        failed = int((group["success"] == False).sum())

        # Laplace smoothing prevents extreme scores
        # when a robot has very few tasks.
        score = (failed + 1) / (total + 2)

        results.append({
            "robot_id": robot_id,
            "reliability_score": score,
        })

    return pd.DataFrame(results)


# ==================================================
# AGENT 3: TASK PATTERN INVESTIGATOR
# ==================================================

def task_pattern_agent(history):
    results = []

    for robot_id, group in history.groupby("robot_id"):

        task_scores = []

        if "task_type" in group.columns:

            for _, task_group in group.groupby("task_type"):

                if len(task_group) >= 3:

                    failures = int(
                        (task_group["success"] == False).sum()
                    )

                    score = (
                        failures + 1
                    ) / (
                        len(task_group) + 2
                    )

                    task_scores.append(score)

        results.append({
            "robot_id": robot_id,
            "task_pattern_score": (
                max(task_scores) if task_scores else 0.0
            ),
        })

    return pd.DataFrame(results)


# ==================================================
# NORMALIZE AGENT SCORES
# ==================================================

def normalize_scores(series):

    return series.rank(
        method="average",
        pct=True
    )


# ==================================================
# CONSENSUS COORDINATOR
# ==================================================

def diagnose_collaboratively(history):

    required = {"robot_id", "success"}

    missing = required - set(history.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if history.empty:
        raise ValueError("Execution history is empty.")

    history = history.copy()

    statistical = statistical_agent(history)
    reliability = reliability_agent(history)
    patterns = task_pattern_agent(history)

    results = statistical.merge(
        reliability,
        on="robot_id"
    ).merge(
        patterns,
        on="robot_id"
    )

    task_summary = history.groupby("robot_id").agg(
        total_tasks=("success", "count"),
        failed_tasks=("success", lambda x: (x == False).sum()),
    ).reset_index()

    results = results.merge(
        task_summary,
        on="robot_id"
    )

    results["failure_rate"] = (
        results["failed_tasks"]
        / results["total_tasks"]
    )

    # Independent normalized agent assessments.
    results["statistical_vote"] = normalize_scores(
        results["statistical_score"]
    )

    results["reliability_vote"] = normalize_scores(
        results["reliability_score"]
    )

    results["task_pattern_vote"] = normalize_scores(
        results["task_pattern_score"]
    )

    # Equal-weight consensus.
    results["consensus_score"] = results[
        [
            "statistical_vote",
            "reliability_vote",
            "task_pattern_vote",
        ]
    ].mean(axis=1)

    results = results.sort_values(
        "consensus_score",
        ascending=False
    ).reset_index(drop=True)

    results.insert(
        0,
        "diagnostic_rank",
        range(1, len(results) + 1)
    )

    return results


# ==================================================
# DIAGNOSTIC EXPLANATIONS
# ==================================================

def explain_consensus(results):

    explanations = []

    for _, row in results.iterrows():

        explanation = (
            f"{row['robot_id']} ranked #{int(row['diagnostic_rank'])}. "
            f"It experienced {int(row['failed_tasks'])} failures "
            f"across {int(row['total_tasks'])} tasks. "
            f"Failure rate: {row['failure_rate']:.2%}. "
            f"Statistical vote: {row['statistical_vote']:.3f}. "
            f"Reliability vote: {row['reliability_vote']:.3f}. "
            f"Task-pattern vote: {row['task_pattern_vote']:.3f}. "
            f"Final consensus: {row['consensus_score']:.3f}."
        )

        explanations.append({
            "robot_id": row["robot_id"],
            "diagnostic_rank": int(row["diagnostic_rank"]),
            "explanation": explanation,
        })

    return pd.DataFrame(explanations)
