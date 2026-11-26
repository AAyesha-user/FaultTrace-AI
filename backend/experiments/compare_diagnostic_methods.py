
from pathlib import Path

import pandas as pd

from backend.diagnosis.suspiciousness import calculate_ochiai
from backend.diagnosis.collaborative import diagnose_collaboratively
from backend.diagnosis.mrsd import diagnose_mrsd


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "mrsd_validation"

HISTORY_FILE = DATA / "execution_history.csv"

OUTPUT = DATA / "method_comparison"

OUTPUT.mkdir(parents=True, exist_ok=True)

ACTUAL_FAULTS = {"R3"}


# ==========================================
# LOAD EXPERIMENTAL DATA
# ==========================================

print("\n" + "=" * 60)
print("     FAULTTRACE AI - DIAGNOSTIC COMPARISON")
print("=" * 60)

if not HISTORY_FILE.exists():
    raise FileNotFoundError(
        f"Missing experimental history: {HISTORY_FILE}"
    )

history = pd.read_csv(HISTORY_FILE)

print("Execution records:", len(history))
print("Experimental runs:", history["run_id"].nunique())


# ==========================================
# HELPER: EXTRACT RANKINGS
# ==========================================

def get_rankings(result, preferred_columns):

    if result is None or result.empty:
        return []

    score_column = next(
        (
            column
            for column in preferred_columns
            if column in result.columns
        ),
        None
    )

    if score_column is None:
        raise ValueError(
            f"No supported diagnostic score found. "
            f"Available columns: {list(result.columns)}"
        )

    ordered = result.sort_values(
        score_column,
        ascending=False
    )

    return ordered["robot_id"].tolist()


# ==========================================
# EVALUATION
# ==========================================

results = []

print("\nStarting three-engine comparison...\n")

for run_id, run_data in history.groupby("run_id"):

    print(f"Evaluating {run_id}...")

    # BASELINE OCHIAI

    ochiai = calculate_ochiai(run_data)

    ochiai_ranking = get_rankings(
        ochiai,
        ["suspiciousness", "ochiai_score"]
    )

    # COLLABORATIVE AI

    collaborative = diagnose_collaboratively(
        run_data
    )

    collaborative_ranking = get_rankings(
        collaborative,
        ["consensus_score"]
    )

    # MRSD

    mrsd = diagnose_mrsd(run_data)

    mrsd_ranking = get_rankings(
        mrsd,
        ["diagnostic_score"]
    )

    methods = {
        "Baseline Ochiai": ochiai_ranking,
        "Collaborative AI": collaborative_ranking,
        "MRSD": mrsd_ranking,
    }

    for method_name, ranking in methods.items():

        top1 = (
            bool(ranking)
            and ranking[0] in ACTUAL_FAULTS
        )

        top3 = bool(
            ACTUAL_FAULTS.intersection(ranking[:3])
        )

        reciprocal_rank = 0.0

        for rank, robot in enumerate(
            ranking, start=1
        ):

            if robot in ACTUAL_FAULTS:

                reciprocal_rank = 1.0 / rank
                break

        results.append({
            "run_id": run_id,
            "method": method_name,
            "top_ranked_robot": (
                ranking[0] if ranking else None
            ),
            "actual_fault_identified": (
                bool(ACTUAL_FAULTS.intersection(ranking))
            ),
            "top1": int(top1),
            "top3": int(top3),
            "reciprocal_rank": reciprocal_rank,
        })


# ==========================================
# COMPARISON REPORT
# ==========================================

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUTPUT / "per_run_comparison.csv",
    index=False
)

summary = (
    comparison.groupby("method")
    .agg(
        top1_accuracy=("top1", "mean"),
        top3_accuracy=("top3", "mean"),
        mean_reciprocal_rank=("reciprocal_rank", "mean"),
        fault_identification_rate=(
            "actual_fault_identified", "mean"
        ),
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT / "method_summary.csv",
    index=False
)


# ==========================================
# FINAL RESULTS
# ==========================================

print("\n" + "=" * 60)
print("          THREE-ENGINE COMPARISON")
print("=" * 60)

print(
    summary.to_string(
        index=False,
        formatters={
            "top1_accuracy": "{:.2%}".format,
            "top3_accuracy": "{:.2%}".format,
            "mean_reciprocal_rank": "{:.4f}".format,
            "fault_identification_rate": "{:.2%}".format,
        }
    )
)

print("\nReports saved inside:")
print(OUTPUT)

print("\nDIAGNOSTIC COMPARISON COMPLETED.")
