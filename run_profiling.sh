#!/bin/bash
# Script to run the data loading profiler with specified parameters.

set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting Data Loading Profiling..."
echo "Current directory: $(pwd)"

# --- Conda Environment Activation ---
# Try to find and initialize conda
CONDA_BASE=""
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
fi

if [ -n "$CONDA_BASE" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Warning: Could not find a standard conda initialization script."
fi

# Activate the environment if it's not already active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "aegis" ]; then
    echo "Activating conda environment: aegis"
    eval "$(conda shell.bash hook)"
    conda activate aegis || {
        echo "Error: Failed to activate conda environment 'aegis'."
        echo "Please ensure the environment exists: conda env list"
        exit 1
    }
fi
echo "Conda environment 'aegis' is active. Python path: $(which python)"
# --- End Conda Activation ---


# Verify the profiler script exists
if [ ! -f "profile_data_loading.py" ]; then
    echo "Error: profile_data_loading.py not found in the current directory."
    echo "Please ensure you are in the project root."
    exit 1
fi

echo "Launching profiler..."
# Run the profiler with arguments from your training script
# The profiler will test both HDF5 and .pt loading internally.
# Since you only use HDF5, you can ignore the .pt results in the final summary.
python -u profile_data_loading.py \
    --data_root_dir /mnt/e/features_uni_v2 \
    --dataset_csv Data/tcga-ot_train.csv \
    --label_col OncoTreeCode \
    --patient_id_col case_id \
    --slide_id_col slide_id \
    --task multiclass \
    --task_type classification \
    --backbone uni_v2 \
    --in_dim 1536 \
    --n_subsamples 2048 \
    --batch_size 16 \
    --profile_num_workers 0 4 8 12 16 \
    --profile_epochs 2 \
    --exp_code tcga_ot_multiclass_s1 \
    --seed 42 \
    --num_workers 16 \
    --log_data \
    --testing \
    --k 1 \
    --k_start 0 \
    --k_end 1 \
    --model_type att_mil \
    --backbone uni_v2 \
    --in_dim 1536 \
    --max_epochs 200 \
    --lr 1e-4 \
    --reg 1e-5 \
    --opt adam \
    --drop_out 0.25 \
    --early_stopping \
    --weighted_sample \
    --batch_size 16 \
    --use_hdf5 \
    --n_subsamples 2048 \
    --preloading no

echo "Profiling finished."

