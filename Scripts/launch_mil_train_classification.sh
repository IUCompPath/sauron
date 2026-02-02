#!/bin/bash
# This script launches the aegis MIL training job for CLASSIFICATION tasks.
# It demonstrates the use of multi-modal metadata fusion.

set -euo pipefail

# --- Configuration ---
# Task and Experiment
TASK_NAME="TCGA_NSCLC_subtype" # e.g., TCGA_NSCLC_subtype, BRACS_tumor_type
TASK_TYPE="classification"     # "classification" or "survival"
EXP_CODE_PREFIX="Fusion_Experiments"

# Data Directories
DATA_ROOT_DIR="/path/to/your/datasets/TCGA-NSCLC" # Change to your data root
RESULTS_DIR="./experiments/training_results"
SPLITS_DIR_BASE="./splits" # Base directory for split files

# CSV column names (must match your dataset CSV)
LABEL_COL="label"           # e.g. "OncoTreeCode" for TCGA-OT, "label" otherwise
PATIENT_ID_COL="case_id"
SLIDE_ID_COL="slide_id"

# --- Multi-modal Metadata Configuration ---
# Comma-separated list of columns from your CSV to use as side-information.
# The model will use the "Concatenation + Projection" strategy to fuse these.
# Example: "age,sex,smoking_status" or "OncoTreeSiteCode"
METADATA_COLS="age,sex" 

# Model Configuration
MODEL_TYPE='att_mil' # 'att_mil', 'trans_mil', 'mamba_mil', 'clam_sb', 'clam_mb'
BACKBONE='titan'    # 'resnet50', 'plip', 'titan'
IN_DIM=1024            # 1024 for titan/resnet50, 512 for plip

# MambaMIL Specific Params (if MODEL_TYPE is mamba_mil)
MAMBA_TYPE="SRMamba"   # 'Mamba', 'BiMamba', 'SRMamba'
MAMBA_LAYERS=2
MAMBA_RATE=5

# Training Hyperparameters
MAX_EPOCHS=50
LR=1e-4
WEIGHT_DECAY=1e-4
OPTIMIZER='adam'
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
    "--split_dir" "${SPLITS_DIR_BASE}/${TASK_NAME}_kfold"
)

# Multi-modal: add metadata columns for classification
if [ -n "${METADATA_COLS:-}" ]; then
    ARGS+=("--metadata_cols" "$METADATA_COLS")
    # Note: The BaseMILModel now uses "Concatenation + Projection" for robust fusion
fi

echo "========================================================================"
echo "Starting aegis MIL Training Job (Classification)"
echo "Experiment Code: $EXP_CODE"
echo "Task: $TASK_NAME"
echo "Model: $MODEL_TYPE"
echo "Metadata Columns: ${METADATA_COLS:-None}"
echo "------------------------------------------------------------------------"
echo "Full command:"
echo "aegis-train ${ARGS[@]}"
echo "========================================================================"

# Run the training command
# Ensure you have installed your package with `pip install -e .`
aegis-train "${ARGS[@]}"

echo "------------------------------------------------------------------------"
echo "aegis MIL Training job completed."
