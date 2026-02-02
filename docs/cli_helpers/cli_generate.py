# cli_generate.py

import sys
import os
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def generate_help_text():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cli_script = os.path.join(root_dir, "aegis", "cli.py")
    output_dir = os.path.join(root_dir, "docs", "generated")

    # For extract_patches
    extract_patches_help = subprocess.check_output(
        ["python", cli_script, "extract_patches", "--help"]
    )
    with open(os.path.join(output_dir, "aegis_extract_patches_help.txt"), "w") as f:
        f.write(extract_patches_help.decode("utf-8"))

    # For train_mil
    train_mil_help = subprocess.check_output(
        ["python", cli_script, "train_mil", "--help"]
    )
    with open(os.path.join(output_dir, "aegis_train_mil_help.txt"), "w") as f:
        f.write(train_mil_help.decode("utf-8"))


if __name__ == "__main__":
    generate_help_text()
