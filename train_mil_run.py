import argparse

from aegis.parse.cli_parsers import get_mil_args
from aegis.training.cli_runner import run_mil_training_job


def main():
    """
    Entry point for running MIL training as a standalone script.
    """
    parser = argparse.ArgumentParser(description="Aegis MIL Training")
    get_mil_args(parser)
    args = parser.parse_args()

    if not hasattr(args, "task_name"):
        args.task_name = args.task
    if not hasattr(args, "k_fold"):
        args.k_fold = args.k

    print(f"Launching Aegis MIL Training with arguments: {args}")
    run_mil_training_job(args)
    print("Aegis MIL Training job completed.")


if __name__ == "__main__":
    main()
