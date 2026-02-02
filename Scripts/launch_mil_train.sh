#!/bin/bash
# This script launches the aegis MIL training job.

set -euo pipefail

# --- Configuration ---
# Task and Experiment
TASK_NAME="TCGA_BRCA_survival" # e.g., BRACS, LUAD_LUSC, TCGA_BRCA_survival
TASK_TYPE="survival" # "classification" or "survival"
EXP_CODE_PREFIX="MambaMIL_Experiments"

# Data Directories
DATA_ROOT_DIR="/path/to/your/datasets/TCGA-BRCA" # Change to your data root
RESULTS_DIR="./experiments/training_results"
SPLITS_DIR_BASE="./splits" # Base directory for split files

# CSV column names (must match your dataset CSV)
LABEL_COL="label"           # e.g. "OncoTreeCode" for TCGA-OT, "label" otherwise
PATIENT_ID_COL="case_id"
SLIDE_ID_COL="slide_id"

# Multi-modal metadata (classification): comma-separated CSV columns fused as extra modality
# e.g. "OncoTreeSiteCode" or "OncoTreeSiteCode,sex" (leave empty to disable)
METADATA_COLS=""

# Model Configuration
MODEL_TYPE='att_mil' # 'att_mil', 'trans_mil', 'mamba_mil', etc.
BACKBONE='titan'    # 'resnet50', 'plip'
IN_DIM=1024            # 1024 for resnet50, 512 for plip

# MambaMIL Specific Params
MAMBA_TYPE="SRMamba"   # 'Mamba', 'BiMamba', 'SRMamba'
MAMBA_LAYERS=2
MAMBA_RATE=5

# Training Hyperparameters
MAX_EPOCHS=100
LR=2e-4
WEIGHT_DECAY=1e-5
OPTIMIZER='adamw'
DROPOUT=0.25
BATCH_SIZE=1

# Cross-validation
K_FOLDS=5

# --- Execution ---
# Construct the full experiment code
EXP_CODE="${EXP_CODE_PREFIX}/${TASK_NAME}/${MODEL_TYPE}_${BACKBONE}"

# Construct the command
declare -a ARGS
ARGS+=(
    "--task" "$TASK_NAME"
    "--task_type" "$TASK_TYPE"
    "--data_root_dir" "$DATA_ROOT_DIR"
    "--results_dir" "$RESULTS_DIR"
    "--exp_code" "$EXP_CODE"
    "--model_type" "$MODEL_TYPE"
    "--backbone" "$BACKBONE"
    "--in_dim" "$IN_DIM"
    "--max_epochs" "$MAX_EPOCHS"
    "--lr" "$LR"
    "--reg" "$WEIGHT_DECAY"
    "--opt" "$OPTIMIZER"
    "--drop_out" "$DROPOUT"
    "--batch_size" "$BATCH_SIZE"
    "--k" "$K_FOLDS"
    "--early_stopping"
    "--weighted_sample"
    "--log_data"
    # Mamba Args
    "--mambamil_type" "$MAMBA_TYPE"
    "--mambamil_layer" "$MAMBA_LAYERS"
    "--mambamil_rate" "$MAMBA_RATE"
    # CSV column names
    "--label_col" "$LABEL_COL"
    "--patient_id_col" "$PATIENT_ID_COL"
    "--slide_id_col" "$SLIDE_ID_COL"
    # Specify paths to your data files
    "--dataset_csv" "/path/to/your/${TASK_NAME}.csv"
    "--split_dir" "${SPLITS_DIR_BASE}/${TASK_NAME}_kfold" # Example split dir name
)

# For survival tasks, add survival-specific arguments
if [ "$TASK_TYPE" = "survival" ]; then
    ARGS+=(
        "--bag_loss" "nll_surv"
        "--alpha_surv" "0.5"
    )
fi

# Multi-modal: add metadata columns for classification (e.g. OncoTreeSiteCode)
if [ -n "${METADATA_COLS:-}" ]; then
    ARGS+=("--metadata_cols" "$METADATA_COLS")
fi

echo "========================================================================"
echo "Starting aegis MIL Training Job"
echo "Experiment Code: $EXP_CODE"
echo "Task: $TASK_NAME ($TASK_TYPE)"
echo "Model: $MODEL_TYPE"
echo "Backbone: $BACKBONE"
echo "------------------------------------------------------------------------"
echo "Full command:"
echo "aegis-train ${ARGS[@]}"
echo "========================================================================"

# Run the training command
# Ensure you have installed your package with `pip install -e .`
# so that `aegis-train` is available in your environment.
aegis-train "${ARGS[@]}"

echo "------------------------------------------------------------------------"
echo "aegis MIL Training job completed."