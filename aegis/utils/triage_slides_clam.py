import argparse
import logging
import os
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# --- Configuration ---
# The magnifications we want to generate patches for.
TARGET_MAGNIFICATIONS = ["20x", "10x", "5x", "2.5x"]

# The order of preference for source magnifications to downsample from.
# Always check for the highest fidelity source first.
SOURCE_PREFERENCE = ["40x", "20x", "10x", "5x"]


def get_mag_value(mag_str: str) -> float:
    """Converts magnification string '20x' to a float 20.0"""
    return float(mag_str.replace("x", ""))


def find_best_strategy(slide_row: pd.Series, target_mag: str) -> dict or None:
    """
    Determines the best way to get a target magnification for a given slide.

    Returns a dictionary with the strategy or None if not possible.
    """
    # Strategy 1: Check if the target magnification is available natively at any level.
    for i in range(int(slide_row.get("level_count", 0))):
        level_mag_col = f"level_{i}_magnification"
        if level_mag_col in slide_row and slide_row[level_mag_col] == target_mag:
            logging.debug(
                f"Found native {target_mag} for {slide_row['slide_file']} at level {i}"
            )
            return {
                "source_mag": target_mag,
                "downsample": 1,
                "level": i,
                "method": "native",
            }

    # Strategy 2: If not native, find the best higher magnification to downsample from.
    target_mag_val = get_mag_value(target_mag)

    for source_mag_pref in SOURCE_PREFERENCE:
        source_mag_val = get_mag_value(source_mag_pref)

        # Only consider sources with higher magnification than the target
        if source_mag_val <= target_mag_val:
            continue

        for i in range(int(slide_row.get("level_count", 0))):
            level_mag_col = f"level_{i}_magnification"
            if (
                level_mag_col in slide_row
                and slide_row[level_mag_col] == source_mag_pref
            ):
                downsample_factor = round(source_mag_val / target_mag_val)
                logging.debug(
                    f"Found {source_mag_pref} for {slide_row['slide_file']} to get {target_mag} with ds={downsample_factor}"
                )
                return {
                    "source_mag": source_mag_pref,
                    "downsample": downsample_factor,
                    "level": i,
                    "method": f"downsample_from_{source_mag_pref}",
                }

    logging.warning(
        f"No viable strategy found for {target_mag} for slide {slide_row['slide_file']}"
    )
    return None


def main(csv_path: Path, output_dir: Path):
    if not csv_path.is_file():
        logging.critical(f"CSV report not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting triage for {len(df)} slides...")

    for _, row in df.iterrows():
        slide_path = Path(row["slide_path"])
        if not slide_path.exists():
            logging.warning(f"Slide file not found, skipping: {slide_path}")
            continue

        for target_mag in TARGET_MAGNIFICATIONS:
            strategy = find_best_strategy(row, target_mag)

            if strategy:
                # Create a descriptive directory name
                if strategy["method"] == "native":
                    dir_name = f"{target_mag}_native"
                else:
                    dir_name = f"{target_mag}_from_{strategy['source_mag']}_ds{strategy['downsample']}"

                triage_subdir = output_dir / dir_name
                triage_subdir.mkdir(exist_ok=True)

                # Create the symbolic link
                symlink_path = triage_subdir / slide_path.name
                if not symlink_path.exists():
                    os.symlink(slide_path.resolve(), symlink_path)
                    logging.info(f"Linking {slide_path.name} -> {triage_subdir.name}")

    logging.info("Triage complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Triage WSI files based on available magnifications for CLAM processing."
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to the wsi_detailed_report.csv file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the root directory where triage symlinks will be created.",
    )
    args = parser.parse_args()
    main(args.csv, args.output)
