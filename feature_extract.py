import argparse
from aegis.feature_extraction.cli_runner import run_feature_extraction_job
from aegis.parse.cli_parsers import build_feature_extraction_parser


def main():
    """
    Entry point for running feature extraction as a standalone script.
    """
    parser = argparse.ArgumentParser(description="Aegis Feature Extraction")
    build_feature_extraction_parser(parser)
    args = parser.parse_args()

    print(f"Launching Aegis Feature Extraction with arguments: {args}")
    run_feature_extraction_job(args)
    print("Aegis Feature Extraction job completed.")


if __name__ == "__main__":
    main()
