import argparse
from aegis.feature_extraction.cli_runner import run_feature_extraction_job
from aegis.parse.cli_parsers import build_feature_extraction_parser, get_mil_args
from aegis.training.cli_runner import run_mil_training_job


def main():
    parser = argparse.ArgumentParser(description="Aegis CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Extractor subparser
    extractor_parser = subparsers.add_parser(
        "extract_patches", help="Extract patches from WSIs"
    )
    build_feature_extraction_parser(extractor_parser)
    extractor_parser.set_defaults(func=feature_extract_main)

    # Trainer subparser
    trainer_parser = subparsers.add_parser("train_mil", help="Train a MIL model")
    get_mil_args(trainer_parser)
    trainer_parser.set_defaults(func=train_mil_main)

    args = parser.parse_args()
    args.func(args)


def feature_extract_main(args):
    """
    Entry point for the 'aegis extract_patches' command.
    """
    print(f"Launching aegis Feature Extraction with arguments: {args}")
    run_feature_extraction_job(args)
    print("aegis Feature Extraction job completed.")


def train_mil_main(args):
    """
    Entry point for the 'aegis train_mil' command.
    """
    if not hasattr(args, "task_name"):
        args.task_name = args.task
    if not hasattr(args, "k_fold"):
        args.k_fold = args.k

    print(f"Launching aegis MIL Training with arguments: {args}")
    run_mil_training_job(args)
    print("aegis MIL Training job completed.")


if __name__ == "__main__":
    main()
