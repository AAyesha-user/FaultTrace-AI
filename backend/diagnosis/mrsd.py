
from itertools import combinations

import pandas as pd


def diagnose_mrsd(history: pd.DataFrame, top_k=None) -> pd.DataFrame:
    """
    MRSD-inspired multi-run fault diagnosis.

    Uses execution outcomes and robot participation patterns.
    Does NOT use faulty_behavior or ground-truth fault labels.

    Produces:
    - Ochiai suspiciousness
    - Failure rate
    - Intermittent failure evidence
    - Cross-run consistency
    - Combined diagnostic ranking
    """

    required_columns = {"robot_id", "success"}

    missing = required_columns - set(history.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if history.empty:
        return pd.DataFrame()

    data = history.copy()

    data["success"] = (
        data["success"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    data = data.dropna(subset=["robot_id", "success"])

    if data.empty:
        return pd.DataFrame()

    total_failed = int((~data["success"]).sum())

    results = []

    for robot_id, group in data.groupby("robot_id"):

        total_tasks = len(group)

        failed = int((~group["success"]).sum())

        successful = int(group["success"].sum())

        failure_rate = failed / total_tasks if total_tasks else 0

        if total_failed > 0 and total_tasks > 0:
            ochiai = failed / (
                (total_failed * total_tasks) ** 0.5
            )
        else:
            ochiai = 0.0

        # Evidence that the robot exhibits both
        # successful and unsuccessful behavior.
        intermittent_evidence = (
            1.0 if failed > 0 and successful > 0 else 0.0
        )

        # Measure whether the robot repeatedly appears
        # in different experimental runs.
        if "run_id" in data.columns:

            robot_runs = data[
                data["robot_id"] == robot_id
            ]

            run_count = robot_runs["run_id"].nunique()

            run_scores = []

            for _, run_group in robot_runs.groupby("run_id"):

                run_failures = int(
                    (~run_group["success"]).sum()
                )

                run_scores.append(
                    run_failures / len(run_group)
                )

            cross_run_consistency = (
                sum(score > 0 for score in run_scores)
                / len(run_scores)
                if run_scores else 0.0
            )

        else:
            run_count = 1

            cross_run_consistency = intermittent_evidence

        # Combined diagnostic score.
        # This is our initial project-specific scoring extension,
        # not a claim that these weights reproduce the paper.
        diagnostic_score = (
            0.60 * ochiai
            + 0.25 * failure_rate
            + 0.10 * intermittent_evidence
            + 0.05 * cross_run_consistency
        )

        results.append({
            "robot_id": robot_id,
            "total_tasks": total_tasks,
            "successful_tasks": successful,
            "failed_tasks": failed,
            "failure_rate": failure_rate,
            "ochiai_score": ochiai,
            "intermittent_evidence": intermittent_evidence,
            "runs_observed": run_count,
            "cross_run_consistency": cross_run_consistency,
            "diagnostic_score": diagnostic_score,
        })

    rankings = pd.DataFrame(results)

    rankings = rankings.sort_values(
        "diagnostic_score",
        ascending=False
    ).reset_index(drop=True)

    rankings.insert(
        0,
        "diagnostic_rank",
        range(1, len(rankings) + 1)
    )

    if top_k is not None:
        rankings = rankings.head(top_k)

    return rankings


def generate_candidate_fault_sets(
    rankings: pd.DataFrame,
    max_faults: int = 3
):
    """
    Generate possible combinations of faulty robots
    from the highest-ranked diagnostic candidates.
    """

    if rankings.empty:
        return pd.DataFrame()

    candidates = rankings.head(
        min(5, len(rankings))
    )

    robot_ids = candidates["robot_id"].tolist()

    results = []

    for size in range(1, min(max_faults, len(robot_ids)) + 1):

        for combination in combinations(robot_ids, size):

            scores = candidates[
                candidates["robot_id"].isin(combination)
            ]["diagnostic_score"]

            results.append({
                "candidate_fault_set": ", ".join(combination),
                "number_of_faults": size,
                "combined_score": float(scores.mean()),
            })

    return pd.DataFrame(results).sort_values(
        "combined_score",
        ascending=False
    ).reset_index(drop=True)


def explain_diagnosis(rankings: pd.DataFrame):

    explanations = []

    for _, robot in rankings.iterrows():

        explanations.append({
            "robot_id": robot["robot_id"],
            "rank": int(robot["diagnostic_rank"]),
            "diagnostic_score": round(
                float(robot["diagnostic_score"]), 4
            ),
            "explanation": (
                f"{robot['robot_id']} experienced "
                f"{int(robot['failed_tasks'])} failures across "
                f"{int(robot['total_tasks'])} tasks. "
                f"Its failure rate was "
                f"{robot['failure_rate']:.2%}, "
                f"with Ochiai suspiciousness "
                f"{robot['ochiai_score']:.4f}."
            )
        })

    return pd.DataFrame(explanations)
