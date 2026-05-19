#!/bin/bash
# This script launches the CLAM create_patches_fp.py job for patch extraction and segmentation.

set -euo pipefail

# --- Configuration Variables ---
# Adjust these as needed for your environment and job.

# Directories and Paths
SOURCE_DIR="./CPTAC_COAD/COAD/"
STEP_SIZE="224"
PATCH_SIZE="224"
PATCH=true
SEG=true
STITCH=false
NO_AUTO_SKIP=true
SAVE_DIR="./CPTAC_COAD/COAD-features"
PRESET=""           # e.g., "/path/to/preset.csv" or leave empty
PATCH_LEVEL="0"
PROCESS_LIST=""     # e.g., "/path/to/list.csv" or leave empty

# --- Constructing the Command ---
declare -a ARGS

ARGS+=("--source" "$SOURCE_DIR")
ARGS+=("--step_size" "$STEP_SIZE")
ARGS+=("--patch_size" "$PATCH_SIZE")
$PATCH && ARGS+=("--patch")
$SEG && ARGS+=("--seg")
$STITCH && ARGS+=("--stitch")
$NO_AUTO_SKIP || ARGS+=("--no_auto_skip")
[ -n "$SAVE_DIR" ] && ARGS+=("--save_dir" "$SAVE_DIR")
[ -n "$PRESET" ] && ARGS+=("--preset" "$PRESET")
[ -n "$PATCH_LEVEL" ] && ARGS+=("--patch_level" "$PATCH_LEVEL")
[ -n "$PROCESS_LIST" ] && ARGS+=("--process_list" "$PROCESS_LIST")

echo "Starting CLAM patch extraction/segmentation job..."
echo "Command:"
echo "python create_patches_fp.py ${ARGS[@]}"
echo "--------------------------------------------------"

cd CLAM

python create_patches_fp.py "${ARGS[@]}"

echo "--------------------------------------------------"
echo "CLAM patch extraction/segmentation job completed."co