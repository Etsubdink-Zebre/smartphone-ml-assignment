import argparse

from src.train_classification import run_classification_training
from src.train_regression import run_regression_training

def run_classification() -> None:
    print("Running classification pipeline...")
    metrics = run_classification_training()
    print("Classification metrics:", metrics)


def run_regression() -> None:
    print("Running regression pipeline...")
    metrics = run_regression_training()
    print("Regression metrics:", metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run smartphone ML assignment pipelines."
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "all"],
        default="all",
        help="Select which pipeline to run.",
    )
    args = parser.parse_args()

    if args.task in ("classification", "all"):
        run_classification()

    if args.task in ("regression", "all"):
        run_regression()

    print("Pipeline run finished.")


if __name__ == "__main__":
    main()
