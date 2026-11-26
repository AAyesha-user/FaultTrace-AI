
from pathlib import Path
import json

import pandas as pd

from backend.simulation.engine import WarehouseSimulation

from backend.diagnosis.collaborative import (
    diagnose_collaboratively,
    explain_consensus,
)

from backend.diagnosis.mrsd import (
    diagnose_mrsd,
    generate_candidate_fault_sets,
    explain_diagnosis,
)

from backend.recovery.engine import simulate_recovery


# ==================================================
# FAULTTRACE AI - FULL SYSTEM INTEGRATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = ROOT / "data" / "system_tests"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ACTUAL_FAULTS = ["R3"]

print("\n" + "=" * 65)
print("             FAULTTRACE AI - SYSTEM INTEGRATION")
print("=" * 65)


# ==================================================
# STEP 1 - ROBOT SIMULATION
# ==================================================

print("\n[1/6] Starting warehouse simulation...")

simulation = WarehouseSimulation(
    num_robots=5,
    seed=42,
    faulty_robot_ids=ACTUAL_FAULTS,
    fault_probability=0.30
)

simulation.generate_tasks(num_tasks=200)

records = simulation.run()

history = pd.DataFrame(
    [record.to_dict() for record in records]
)

history.to_csv(
    REPORT_DIR / "execution_history.csv",
    index=False
)

print("Simulation completed.")
print("Tasks executed:", len(history))
print("Robots:", history["robot_id"].nunique())


# ==================================================
# STEP 2 - EXECUTION ANALYSIS
# ==================================================

print("\n[2/6] Analyzing execution performance...")

history["success"] = (
    history["success"]
    .astype(str)
    .str.lower()
    .map({
        "true": True,
        "false": False,
        "1": True,
        "0": False
    })
)

history = history.dropna(
    subset=["robot_id", "success"]
)

successful_tasks = int(history["success"].sum())

failed_tasks = int((~history["success"]).sum())

print("Successful tasks:", successful_tasks)
print("Failed tasks:", failed_tasks)

print(
    "Fleet success rate:",
    f"{successful_tasks / len(history):.2%}"
)


# ==================================================
# STEP 3 - COLLABORATIVE DIAGNOSIS
# ==================================================

print("\n[3/6] Activating collaborative diagnostic agents...")

collaborative_results = diagnose_collaboratively(history)

collaborative_explanations = explain_consensus(
    collaborative_results
)

collaborative_results.to_csv(
    REPORT_DIR / "collaborative_diagnosis.csv",
    index=False
)

collaborative_explanations.to_csv(
    REPORT_DIR / "collaborative_explanations.csv",
    index=False
)

print("\nCOLLABORATIVE DIAGNOSTIC RANKING")

print(
    collaborative_results.to_string(index=False)
)


# ==================================================
# STEP 4 - MRSD-INSPIRED DIAGNOSIS
# ==================================================

print("\n[4/6] Activating MRSD diagnostic engine...")

mrsd_results = diagnose_mrsd(history)

mrsd_candidates = generate_candidate_fault_sets(
    mrsd_results,
    max_faults=3
)

mrsd_explanations = explain_diagnosis(
    mrsd_results
)

mrsd_results.to_csv(
    REPORT_DIR / "mrsd_diagnostic_ranking.csv",
    index=False
)

mrsd_candidates.to_csv(
    REPORT_DIR / "mrsd_candidate_fault_sets.csv",
    index=False
)

mrsd_explanations.to_csv(
    REPORT_DIR / "mrsd_explanations.csv",
    index=False
)

print("\nMRSD DIAGNOSTIC RANKING")

print(
    mrsd_results.to_string(index=False)
)

print("\nPOSSIBLE FAULT COMBINATIONS")

if not mrsd_candidates.empty:
    print(
        mrsd_candidates.head(10).to_string(index=False)
    )
else:
    print("No candidate combinations generated.")


# ==================================================
# STEP 5 - AUTONOMOUS RECOVERY
# ==================================================

print("\n[5/6] Activating autonomous recovery engine...")

recovered, recovery_summary = simulate_recovery(
    history,
    ACTUAL_FAULTS
)

recovered.to_csv(
    REPORT_DIR / "recovery_results.csv",
    index=False
)

print("\nRECOVERY SUMMARY")

for key, value in recovery_summary.items():
    print(f"{key}: {value}")


# ==================================================
# STEP 6 - FINAL SYSTEM REPORT
# ==================================================

print("\n[6/6] Generating final system report...")

collaborative_top = (
    collaborative_results.iloc[0]["robot_id"]
    if not collaborative_results.empty
    else None
)

mrsd_top = (
    mrsd_results.iloc[0]["robot_id"]
    if not mrsd_results.empty
    else None
)

report = {
    "system": "FaultTrace AI",

    "robots": int(history["robot_id"].nunique()),

    "tasks_executed": len(history),

    "successful_tasks": successful_tasks,

    "failed_tasks": failed_tasks,

    "actual_faulty_robots": ACTUAL_FAULTS,

    "collaborative_diagnosis": {
        "top_ranked_robot": collaborative_top,
        "identified_fault": collaborative_top in ACTUAL_FAULTS
        if collaborative_top else False,
        "ranking_rows": len(collaborative_results)
    },

    "mrsd_diagnosis": {
        "top_ranked_robot": mrsd_top,
        "identified_fault": mrsd_top in ACTUAL_FAULTS
        if mrsd_top else False,
        "ranking_rows": len(mrsd_results),
        "candidate_fault_sets": len(mrsd_candidates)
    },

    "recovery_summary": recovery_summary,

    "recovery_records": len(recovered)
}

with open(
    REPORT_DIR / "system_report.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4,
        default=str
    )


# ==================================================
# FINAL RESULTS
# ==================================================

print("\n" + "=" * 65)
print("               SYSTEM TEST COMPLETED")
print("=" * 65)

print(
    "\nCollaborative diagnosis:",
    "PASS" if report["collaborative_diagnosis"]["identified_fault"]
    else "REVIEW"
)

print(
    "MRSD diagnosis:",
    "PASS" if report["mrsd_diagnosis"]["identified_fault"]
    else "REVIEW"
)

print(
    "Recovery records generated:",
    len(recovered)
)

print("\nAll reports saved inside:")

print(REPORT_DIR)

print("\nFAULTTRACE AI INTEGRATION TEST FINISHED.")

print("=" * 65)
