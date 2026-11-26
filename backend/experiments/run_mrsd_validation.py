
from pathlib import Path

import pandas as pd

from backend.simulation.engine import WarehouseSimulation
from backend.diagnosis.mrsd import (
    diagnose_mrsd,
    generate_candidate_fault_sets,
    explain_diagnosis,
)


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = ROOT / "data" / "mrsd_validation"
OUTPUT.mkdir(parents=True, exist_ok=True)

NUM_RUNS = 30
TASKS_PER_RUN = 100

FAULTY_ROBOTS = ["R3"]
FAULT_PROBABILITY = 0.30


# ==========================================
# MULTI-RUN SIMULATION
# ==========================================

print("\n" + "=" * 60)
print("       FAULTTRACE AI - MRSD RESEARCH VALIDATION")
print("=" * 60)

all_records = []

for run_number in range(1, NUM_RUNS + 1):

    simulation = WarehouseSimulation(
        num_robots=5,
        seed=42 + run_number,
        faulty_robot_ids=FAULTY_ROBOTS,
        fault_probability=FAULT_PROBABILITY,
    )

    simulation.generate_tasks(
        num_tasks=TASKS_PER_RUN
    )

    records = simulation.run()

    run_records = [
        record.to_dict() for record in records
    ]

    for record in run_records:
        record["run_id"] = f"RUN_{run_number:02d}"

    all_records.extend(run_records)

    print(
        f"Completed run {run_number}/{NUM_RUNS}"
    )


history = pd.DataFrame(all_records)

history.to_csv(
    OUTPUT / "execution_history.csv",
    index=False
)


# ==========================================
# AGGREGATED MRSD DIAGNOSIS
# ==========================================

print("\n[1/4] Running cross-run MRSD diagnosis...")

rankings = diagnose_mrsd(history)

rankings.to_csv(
    OUTPUT / "mrsd_rankings.csv",
    index=False
)

print("\nAGGREGATED DIAGNOSTIC RANKING")
print(rankings.to_string(index=False))


# ==========================================
# CANDIDATE FAULT COMBINATIONS
# ==========================================

print("\n[2/4] Generating candidate fault combinations...")

candidates = generate_candidate_fault_sets(
    rankings,
    max_faults=3
)

candidates.to_csv(
    OUTPUT / "candidate_fault_sets.csv",
    index=False
)

print("\nCANDIDATE FAULT SETS")
print(candidates.head(10).to_string(index=False))


# ==========================================
# EXPLANATIONS
# ==========================================

print("\n[3/4] Generating diagnostic explanations...")

explanations = explain_diagnosis(rankings)

explanations.to_csv(
    OUTPUT / "diagnostic_explanations.csv",
    index=False
)


# ==========================================
# RESEARCH EVALUATION
# ==========================================

print("\n[4/4] Evaluating research performance...")

run_results = []

for run_id, run_data in history.groupby("run_id"):

    run_ranking = diagnose_mrsd(run_data)

    ordered_robots = run_ranking[
        "robot_id"
    ].tolist()

    actual_faults = set(FAULTY_ROBOTS)

    top1 = (
        len(ordered_robots) > 0
        and ordered_robots[0] in actual_faults
    )

    top3 = bool(
        actual_faults.intersection(
            ordered_robots[:3]
        )
    )

    reciprocal_rank = 0.0

    for rank, robot in enumerate(
        ordered_robots, start=1
    ):
        if robot in actual_faults:
            reciprocal_rank = 1.0 / rank
            break

    run_results.append({
        "run_id": run_id,
        "top1": int(top1),
        "top3": int(top3),
        "reciprocal_rank": reciprocal_rank,
    })


evaluation = pd.DataFrame(run_results)

evaluation.to_csv(
    OUTPUT / "evaluation_per_run.csv",
    index=False
)

summary = {
    "total_runs": NUM_RUNS,
    "tasks_per_run": TASKS_PER_RUN,
    "total_tasks": len(history),
    "actual_faulty_robots": FAULTY_ROBOTS,
    "top1_accuracy": float(evaluation["top1"].mean()),
    "top3_accuracy": float(evaluation["top3"].mean()),
    "mean_reciprocal_rank": float(
        evaluation["reciprocal_rank"].mean()
    ),
    "cross_run_consistency": float(
        rankings.iloc[0]["cross_run_consistency"]
    ) if not rankings.empty else 0.0,
    "top_ranked_robot": (
        rankings.iloc[0]["robot_id"]
        if not rankings.empty else None
    ),
    "fault_identified": (
        not rankings.empty
        and rankings.iloc[0]["robot_id"] in FAULTY_ROBOTS
    ),
}

pd.DataFrame([summary]).to_csv(
    OUTPUT / "research_summary.csv",
    index=False
)


# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 60)
print("          MRSD RESEARCH VALIDATION REPORT")
print("=" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nFiles saved inside:")
print(OUTPUT)

print("\nMRSD RESEARCH VALIDATION COMPLETED.")
