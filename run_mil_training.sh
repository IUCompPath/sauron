#!/bin/bash
# Script to run MIL training with specified parameters
#
# NOTE: If using mambamil model, ensure mamba-ssm is installed:
#   pip install mamba-ssm
#   or
#   pip install causal-conv1d>=1.2.0
#   (CUDA extensions will be compiled automatically)
#
# Available MIL model types (--model_type):
#   - att_mil        : Attention-based MIL (ABMIL/DAttention)
#   - trans_mil       : Transformer-based MIL (TransMIL)
#   - max_mil         : Max pooling MIL
#   - mean_mil        : Mean pooling MIL
#   - s4model         : S4-based MIL
#   - wikgmil         : WiKG (Graph-based MIL)
#   - diffabmil       : Differentiable Attention MIL
#   - hgachc          : Hierarchical Graph Attention Cross-Head Communication
#   - rrtmil          : RRT (Region-based Transformer MIL)
#   - dsmil           : Dual-stream MIL
#   - mambamil        : Mamba-based MIL (supports SRMamba, Mamba, BiMamba)
#   - moemil          : Mixture of Experts MIL
#
# Model-specific parameters:
#   - mambamil: --mamba_layers (default: 2), --mamba_rate (default: 10), --mamba_type (SRMamba/Mamba/BiMamba)
#   - moemil: --embed_dim (default: 512), --num_experts (default: 4)
#   - trans_mil, att_mil, max_mil, mean_mil, s4model: --activation (relu/gelu, default varies by model)

set -e  # Exit on error

echo "Starting MIL training script..."
echo "Current directory: $(pwd)"

# Initialize conda for bash script
# Only attempt if conda is available (skip in Docker if using system python)
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")

    if [ -n "$CONDA_BASE" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi

    # Activate conda environment
    echo "Activating conda environment: aegis"
    # Initialize conda for this shell
    eval "$(conda shell.bash hook)"
    conda activate aegis || {
        echo "Warning: Failed to activate conda environment 'aegis'. Continuing with current python..."
    }
else
    echo "Conda not found. Assuming environment is already configured (e.g. Docker)."
fi
echo "Conda environment activated. Python path: $(which python)"

# Verify Python script exists
if [ ! -f "train_mil_run.py" ]; then
    echo "Error: train_mil_run.py not found in current directory: $(pwd)"
    echo "Please ensure you're running this script from the project root directory"
    exit 1
fi

# Set DATA_ROOT based on OS
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Native Windows (Git Bash, etc.): use Windows path format
    DATA_ROOT="${DATA_ROOT:-E:\features_uni_v2}"
else
    # Linux/Unix/WSL: use /mnt/e mount point
    DATA_ROOT="${DATA_ROOT:-/mnt/e/features_uni_v2}"
fi

echo "Launching training..."

# Optional: Set memmap paths if you have converted your data to memmap format
# To use memmap datasets for faster I/O, first convert your H5 files:
#   python -u aegis/utils/convert_to_memmap.py \
#       --source_dir /path/to/h5_files \
#       --output_bin /path/to/dataset.bin \
#       --output_json /path/to/dataset_index.json
#
# Then set these environment variables before running this script:
#   export MEMMAP_BIN_PATH="/path/to/dataset.bin"
#   export MEMMAP_JSON_PATH="/path/to/dataset_index.json"
# Or uncomment and set them directly below:
# MEMMAP_BIN_PATH="${MEMMAP_BIN_PATH:-/path/to/dataset.bin}"
# MEMMAP_JSON_PATH="${MEMMAP_JSON_PATH:-/path/to/dataset_index.json}"

# Build base command arguments
PYTHON_ARGS=(
    --data_root_dir "/data/features_uni_v2"
    --train_csv Data/tcga-ot_train.csv
    --val_csv Data/tcga-ot_val.csv
    --test_csv Data/tcga-ot_test.csv
    --label_col OncoTreeCode
    --patient_id_col case_id
    --slide_id_col slide_id
    --results_dir ./results
    --task multiclass
    --task_type classification
    --exp_code tcga_ot_multiclass_s1
    --seed 42
    --num_workers 16
    --log_data
    --testing
    --k 1
    --k_start 0
    --k_end 1
    --model_type att_mil
    --backbone uni_v2
    --in_dim 1536
    --max_epochs 200
    --lr 1e-4
    --reg 1e-5
    --opt adam
    --drop_out 0.25
    --early_stopping
    --preloading no
    --weighted_sample
    --batch_size 8
    --use_hdf5
    --n_subsamples 2048
    --loss_type poly
    # --memmap_bin_path "E:\\dataset.bin"
    # --memmap_json_path "E:\\output.json"
)

# Add memmap arguments if both paths are provided
# if [ -n "${MEMMAP_BIN_PATH:-}" ] && [ -n "${MEMMAP_JSON_PATH:-}" ]; then
#     PYTHON_ARGS+=(--memmap_bin_path "$MEMMAP_BIN_PATH")
#     PYTHON_ARGS+=(--memmap_json_path "$MEMMAP_JSON_PATH")
#     echo "Using memmap datasets:"
#     echo "  Binary file: $MEMMAP_BIN_PATH"
#     echo "  Index file: $MEMMAP_JSON_PATH"
# else
#     echo "Using standard HDF5/PT file datasets (set MEMMAP_BIN_PATH and MEMMAP_JSON_PATH to use memmap)"
# fi

python -u train_mil_run.py "${PYTHON_ARGS[@]}"

# Example: To use a different model, change --model_type:
# --model_type trans_mil --activation gelu
# --model_type att_mil --activation relu
# --model_type moemil --embed_dim 512 --num_experts 4
# --model_type mambamil --mamba_layers 2 --mamba_rate 10 --mamba_type SRMamba
