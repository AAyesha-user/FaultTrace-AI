from pathlib import Path
import pandas as pd

from backend.evaluation.metrics import evaluate_diagnosis


def main():
    data_dir = Path("data/experiments")

    rankings = pd.read_csv(
        data_dir / "diagnostic_rankings.csv"
    )

    ground_truth = pd.read_csv(
        data_dir / "ground_truth.csv"
    )

    results, summary = evaluate_diagnosis(
        rankings, ground_truth
    )

    results.to_csv(
        data_dir / "evaluation_per_run.csv",
        index=False,
    )

    print("\n=== FaultTrace AI: Evaluation Report ===")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Top-1 accuracy: {summary['top1_accuracy']:.2%}")
    print(f"Top-3 accuracy: {summary['top3_accuracy']:.2%}")
    print(f"Mean Reciprocal Rank: {summary['mrr']:.4f}")

    print("\nIndividual run results:")
    print(results.head(10).to_string(index=False))

    print("\nSaved:")
    print(data_dir / "evaluation_per_run.csv")


if __name__ == "__main__":
    main()