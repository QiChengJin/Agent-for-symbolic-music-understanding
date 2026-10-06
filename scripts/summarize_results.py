"""Recalculate headline accuracies from checked-in experiment outputs."""

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUNS = (
    ("Emotion, direct prediction", "emotion_recognition_results.csv", "prediction"),
    ("Emotion, single analyst", "emotion_recognition_agent_results.csv", "prediction_single"),
    ("Emotion, majority vote", "emotion_recognition_agent_results.csv", "prediction_majority"),
    ("Emotion, agent judge", "emotion_recognition_agent_results.csv", "prediction_agent"),
    ("Metadata QA, Llama baseline", "metadata_QA_baseline_results.csv", "pred"),
    ("Metadata QA, Gemma baseline", "metadata_QA_gemma_baseline_results.csv", "pred"),
)


def accuracy(filename: str, prediction_column: str) -> tuple[int, int]:
    path = PROJECT_ROOT / "results" / filename
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    target_column = "ground_truth" if "ground_truth" in rows[0] else "solution"
    correct = sum(
        str(row[target_column]).strip() == str(row[prediction_column]).strip()
        for row in rows
    )
    return correct, len(rows)


def main() -> None:
    print(f"{'Run':36} {'Correct':>9} {'Accuracy':>10}")
    print("-" * 57)
    for label, filename, prediction_column in RUNS:
        correct, total = accuracy(filename, prediction_column)
        print(f"{label:36} {correct:>3}/{total:<5} {correct / total:>9.1%}")


if __name__ == "__main__":
    main()
