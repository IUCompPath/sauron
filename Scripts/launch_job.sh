#!/bin/bash
# This script launches the aegis feature extraction job.

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error.
# The return value of a pipeline is the value of the last (rightmost) command to exit with a non-zero status, or zero if all commands in the pipeline exit successfully.
set -euo pipefail

# --- Configuration Variables ---
# Modify these variables according to your setup and desired job.

# No need for PYTHON_SCRIPT anymore, as we call the installed command.

# Directories and Paths
JOB_DIR="/media/mydrive/TCGA/TCGA-BRCA-features-seg/"
WSI_DIR="/media/mydrive/TCGA/TCGA-BRCA-FLAT/"
WSI_CACHE="" # Set to "" to disable WSI caching

# GPU & Task
CUDA_VISIBLE_DEVICES="0" # GPU ID(s) to use, e.g., "0" or "0,1"
TASK="seg"               # seg, coords, feat, all, cache

# WSI Discovery and Reader
WSI_EXT=".svs"            # Space-separated list for multiple extensions: "svs tif ndpi"
CUSTOM_MPP_KEYS="mpp"    # Space-separated list for multiple keys: "mpp slide_mpp"
CUSTOM_LIST_OF_WSIS=""   # Path to a CSV file (e.g., "/path/to/my_wsis.csv"). Leave empty "" to use all WSIs in WSI_DIR.
READER_TYPE=""           # Force WSI reader: openslide, image, cucim. Leave empty "" for auto-determination.
SEARCH_NESTED="false"    # Set to "true" to recursively search WSI_DIR subdirectories

# Parallelization & Batching
MAX_WORKERS="32"          # Maximum number of workers. Set to "" for inferred (based on CPU cores).
BATCH_SIZE="64"          # General batch size. Can be overridden by seg_batch_size/feat_batch_size.

# Caching Options
CLEAR_CACHE="false"      # Set to "true" to delete cached WSIs after processing each batch.
CACHE_BATCH_SIZE="32"    # Max number of slides to cache locally at once when --wsi_cache is used.

# Error Handling
SKIP_ERRORS="false"      # Set to "true" to skip errored slides and continue processing.

# Segmentation Arguments
SEGMENTER="clam"         # hest, grandqc
SEG_CONF_THRESH="0.5"    # Confidence threshold for segmentation.   
REMOVE_HOLES="false"     # Set to "true" to remove holes from segmentation mask.
REMOVE_ARTIFACTS="false" # Set to "true" to remove artifacts using GrandQC.
REMOVE_PENMARKS="false"  # Set to "true" to remove penmarks specifically (if REMOVE_ARTIFACTS is false).
SEG_BATCH_SIZE=""        # Batch size for segmentation. Set to "" to use BATCH_SIZE.

# Patching Arguments
MAG="20"                 # Magnification level for patching (e.g., 20 for 20x).
PATCH_SIZE="256"         # Side length of square patches in pixels.
OVERLAP="0"              # Absolute overlap between adjacent patches in pixels.
MIN_TISSUE_PROPORTION="0.0" # Minimum proportion of patch area that must contain tissue.
COORDS_DIR_NAME=""       # Name of directory to save/restore coordinates. Set to "" for auto-generated.

# Feature Extraction Arguments
PATCH_ENCODER="conch_v15"       # Patch encoder model (e.g., resnet50, conch_v15). See --help for choices.
PATCH_ENCODER_CKPT_PATH=""      # Local path to a custom checkpoint. Set to "" to use default registry.
SLIDE_ENCODER=""                # Slide encoder model (e.g., threads, mean-virchow). Set to "" to skip slide-level features.
FEAT_BATCH_SIZE=""              # Batch size for feature extraction. Set to "" to use BATCH_SIZE.

# --- Constructing the Command ---
declare -a ARGS

# Required arguments (or always included)
ARGS+=(
    "--job_dir" "$JOB_DIR"
    "--wsi_dir" "$WSI_DIR"
    "--task" "$TASK"
    "--batch_size" "$BATCH_SIZE"
    "--mag" "$MAG"
    "--min_tissue_proportion" "$MIN_TISSUE_PROPORTION"
    "--patch_encoder" "$PATCH_ENCODER"
    "--segmenter" "$SEGMENTER"
)

if [ "$SEGMENTER" != "clam" ]; then
    ARGS+=("--patch_size" "$PATCH_SIZE" "--overlap" "$OVERLAP")
fi

# Optional arguments (only add if their value is not empty or if boolean "true")
[ -n "$WSI_EXT" ] && ARGS+=("--wsi_ext" $WSI_EXT) # No quotes for $WSI_EXT here to allow shell word splitting for nargs
[ -n "$WSI_CACHE" ] && ARGS+=("--wsi_cache" "$WSI_CACHE")
[ "$CLEAR_CACHE" = "true" ] && ARGS+=("--clear_cache")
[ -n "$CACHE_BATCH_SIZE" ] && ARGS+=("--cache_batch_size" "$CACHE_BATCH_SIZE")
[ "$SKIP_ERRORS" = "true" ] && ARGS+=("--skip_errors")
[ -n "$CUSTOM_MPP_KEYS" ] && ARGS+=("--custom_mpp_keys" $CUSTOM_MPP_KEYS) # No quotes for $CUSTOM_MPP_KEYS to allow shell word splitting for nargs
[ -n "$CUSTOM_LIST_OF_WSIS" ] && ARGS+=("--custom_list_of_wsis" "$CUSTOM_LIST_OF_WSIS")
[ -n "$MAX_WORKERS" ] && ARGS+=("--max_workers" "$MAX_WORKERS")
[ "$SEARCH_NESTED" = "true" ] && ARGS+=("--search_nested")
[ -n "$READER_TYPE" ] && ARGS+=("--reader_type" "$READER_TYPE")

# Segmentation specific arguments
[ -n "$SEG_BATCH_SIZE" ] && ARGS+=("--seg_batch_size" "$SEG_BATCH_SIZE")
[ -n "$SEG_CONF_THRESH" ] && ARGS+=("--seg_conf_thresh" "$SEG_CONF_THRESH")
[ "$REMOVE_HOLES" = "true" ] && ARGS+=("--remove_holes")
[ "$REMOVE_ARTIFACTS" = "true" ] && ARGS+=("--remove_artifacts")
[ "$REMOVE_PENMARKS" = "true" ] && ARGS+=("--remove_penmarks")

# Patching/Coordinates specific arguments
[ -n "$COORDS_DIR_NAME" ] && ARGS+=("--coords_dir_name" "$COORDS_DIR_NAME")

# Feature extraction specific arguments
[ -n "$PATCH_ENCODER_CKPT_PATH" ] && ARGS+=("--patch_encoder_ckpt_path" "$PATCH_ENCODER_CKPT_PATH")
[ -n "$SLIDE_ENCODER" ] && ARGS+=("--slide_encoder" "$SLIDE_ENCODER")
[ -n "$FEAT_BATCH_SIZE" ] && ARGS+=("--feat_batch_size" "$FEAT_BATCH_SIZE")

# --- Execution ---
# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

echo "Starting aegis feature extraction job via installed command..."
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Job Directory: $JOB_DIR"
echo "WSI Directory: $WSI_DIR"
echo "Full command being executed:"
echo "aegis-extract ${ARGS[@]}"
echo "--------------------------------------------------"

# Execute the installed aegis command with the constructed arguments
aegis-extract "${ARGS[@]}"

echo "--------------------------------------------------"
echo "aegis Feature Extraction job completed."