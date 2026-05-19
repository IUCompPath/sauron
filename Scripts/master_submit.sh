#!/bin/bash
# This master script orchestrates the triage and parallel submission of CLAM jobs on SLURM.

set -euo pipefail

# --- Configuration Variables ---

# 1. Paths and Directories
CSV_REPORT="/path/to/your/wsi_detailed_report.csv"
TRIAGE_ROOT="./triage_workspace" # A temporary directory for symlinks
SAVE_DIR_BASE="./CPTAC_COAD/COAD-features" # Base output directory
CLAM_DIR="./CLAM" # Path to the CLAM repository
TASK_MAP_FILE="./clam_task_map.txt" # A temporary file listing all jobs

# 2. Desired Output Patch Sizes
OUTPUT_PATCH_SIZES=("224" "256" "512")

# 3. CLAM Script Preset
PRESET="tcga.csv" # e.g., "bwh_resection.csv", or "" to use CLAM defaults

# 4. SLURM Job Array Configuration
MAX_SIMULTANEOUS_JOBS=50 # Controls parallelism. Don't set too high!

# --- SCRIPT LOGIC ---

# 1. Triage Slides (Serial Step)
echo "### STAGE 1: TRIAGING SLIDES ###"
if [ -d "$TRIAGE_ROOT" ]; then
    echo "Triage directory '$TRIAGE_ROOT' already exists. Reusing it."
else
    python3 triage_slides.py --csv "$CSV_REPORT" --output "$TRIAGE_ROOT"
fi
echo "Triage complete. Symlinks are in $TRIAGE_ROOT"
echo "--------------------------------------------------"

# 2. Generate Task Map File (Serial Step)
echo "### STAGE 2: GENERATING TASK MAP ###"
# Clear previous task map if it exists
rm -f "$TASK_MAP_FILE"

task_count=0
# Iterate over each triage directory (e.g., 20x_native, 10x_from_40x_ds4, etc.)
for source_dir in "$TRIAGE_ROOT"/*; do
    if [ -d "$source_dir" ]; then
        dir_name=$(basename "$source_dir")
        
        # Parse strategy from directory name
        target_mag=$(echo "$dir_name" | cut -d'_' -f1)
        downsample_factor=1 # Default for native
        if [[ $dir_name != *"_native"* ]]; then
            downsample_factor=$(echo "$dir_name" | grep -o 'ds[0-9]*' | cut -c 3-)
        fi

        for output_size in "${OUTPUT_PATCH_SIZES[@]}"; do
            # Write one line per job: source_dir output_size downsample_factor target_mag
            echo "$source_dir $output_size $downsample_factor $target_mag" >> "$TASK_MAP_FILE"
            task_count=$((task_count + 1))
        done
    fi
done

if [ "$task_count" -eq 0 ]; then
    echo "No tasks generated. Triage directory might be empty or something went wrong. Exiting."
    exit 1
fi

echo "Generated $task_count tasks in '$TASK_MAP_FILE'."
echo "--------------------------------------------------"

# 3. Submit SLURM Job Array
echo "### STAGE 3: SUBMITTING SLURM JOB ARRAY ###"
# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Submit the worker script as a job array. The array size is the number of lines in our task map.
# The % operator limits how many tasks run concurrently.
sbatch --array=1-${task_count}%${MAX_SIMULTANEOUS_JOBS} \
    clam_worker.sbatch "$TASK_MAP_FILE" "$SAVE_DIR_BASE" "$CLAM_DIR" "$PRESET"

echo
echo "Successfully submitted SLURM job array with $task_count tasks."
echo "Parallelism is limited to $MAX_SIMULTANEOUS_JOBS concurrent jobs."
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs in the 'slurm_logs' directory."