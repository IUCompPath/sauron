#!/bin/bash
#SBATCH --job-name=clam-patch
#SBATCH --partition=general     # Adjust to your cluster's partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4       # Request more CPUs if CLAM is multi-threaded
#SBATCH --mem=16G               # Adjust memory as needed for large slides
#SBATCH --time=02:00:00         # Adjust time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/clam_%A_%a.out  # Log file for stdout (%A=Job ID, %a=Task ID)
#SBATCH --error=slurm_logs/clam_%A_%a.err   # Log file for stderr

set -euo pipefail

# --- Environment Setup ---
# Load any necessary modules (e.g., python, anaconda). This is cluster-specific.
# module load anaconda
# source activate your_clam_env

# --- Configuration (Passed from master script) ---
TASK_MAP_FILE="$1"
SAVE_DIR_BASE="$2"
CLAM_DIR="$3"
PRESET="$4"

# --- Task Execution ---
# Get the parameters for this specific task ID from the task map file
TASK_PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_MAP_FILE")
if [ -z "$TASK_PARAMS" ]; then
    echo "Error: No task parameters found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Parse the parameters
SOURCE_DIR=$(echo "$TASK_PARAMS" | awk '{print $1}')
OUTPUT_SIZE=$(echo "$TASK_PARAMS" | awk '{print $2}')
DOWNSAMPLE_FACTOR=$(echo "$TASK_PARAMS" | awk '{print $3}')
TARGET_MAG=$(echo "$TASK_PARAMS" | awk '{print $4}')

echo "========================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Processing slides from: ${SOURCE_DIR}"
echo "--------------------------------------------------------"

# Calculate the actual patch size needed for extraction
extraction_size=$((OUTPUT_SIZE * DOWNSAMPLE_FACTOR))

# Define the final save directory for this specific job
save_dir="${SAVE_DIR_BASE}/${TARGET_MAG}/patch_${OUTPUT_SIZE}"
mkdir -p "$save_dir"

# --- Construct and Run the CLAM Command ---
cd "$CLAM_DIR"

declare -a ARGS
ARGS+=("--source" "$SOURCE_DIR")
ARGS+=("--save_dir" "$save_dir")
ARGS+=("--patch_size" "$extraction_size")
ARGS+=("--step_size" "$extraction_size")

# Only add custom_downsample if we are actually downsampling
if [ "$DOWNSAMPLE_FACTOR" -gt 1 ]; then
    ARGS+=("--custom_downsample" "$DOWNSAMPLE_FACTOR")
fi

# Add other boolean flags
ARGS+=("--patch")
ARGS+=("--seg")
# ARGS+=("--stitch") # Stitching is usually off for patching
ARGS+=("--no_auto_skip")

[ -n "$PRESET" ] && ARGS+=("--preset" "$PRESET")

echo "Target Magnification: $TARGET_MAG"
echo "Effective Patch Size: $OUTPUT_SIZE"
echo "Downsample Factor: $DOWNSAMPLE_FACTOR"
echo "Extraction Patch Size: $extraction_size"
echo "Final Save Directory: $save_dir"
echo
echo "Executing command:"
echo "python create_patches_fp.py ${ARGS[@]}"
echo "========================================================"

# Use srun to ensure proper resource allocation and tracking by SLURM
srun python create_patches_fp.py "${ARGS[@]}"

echo "Task ${SLURM_ARRAY_TASK_ID} completed successfully."