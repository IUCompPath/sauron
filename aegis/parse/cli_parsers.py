import argparse


def parse_feature_extraction_arguments():
    parser = argparse.ArgumentParser(description="aegis feature extraction")
    build_feature_extraction_parser(parser)
    return parser.parse_args()


def build_feature_extraction_parser(parser: argparse.ArgumentParser):
    """
    Parse command-line arguments for the aegis feature extraction script.
    """

    # Generic arguments
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index to use for processing tasks."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="seg",
        choices=["seg", "coords", "feat", "all", "cache"],
        help="Task to run: seg (segmentation), coords (save tissue coordinates), feat (extract features), all (run all steps), cache (populate WSI cache only).",
    )
    parser.add_argument(
        "--job_dir", type=str, required=True, help="Directory to store outputs."
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        default=False,
        help="Skip errored slides and continue processing.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Maximum number of workers for data loading (e.g., in DataLoader). If None, inferred based on CPU cores.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used for segmentation and feature extraction. Will be overridden by "
        "`seg_batch_size` and `feat_batch_size` if specified. Defaults to 64.",
    )

    # Caching argument for fast WSI processing
    parser.add_argument(
        "--wsi_cache",
        type=str,
        default=None,
        help="Path to a local cache (e.g., SSD) used to speed up access to WSIs stored on slower drives (e.g., HDD). "
        "If provided, WSIs are copied here before processing.",
    )
    parser.add_argument(
        "--cache_batch_size",
        type=int,
        default=32,
        help="Maximum number of slides to cache locally at once when using --wsi_cache. Helps control disk usage.",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        default=False,
        help="If using --wsi_cache, delete cached WSIs after processing each batch.",
    )

    # Slide-related arguments
    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=True,
        help="Directory containing WSI files (can be nested if --search_nested is used).",
    )
    parser.add_argument(
        "--wsi_ext",
        type=str,
        nargs="+",
        default=None,
        help="List of allowed file extensions for WSI files (e.g., .svs .tif). If None, common extensions are used.",
    )
    parser.add_argument(
        "--custom_mpp_keys",
        type=str,
        nargs="+",
        default=None,
        help="Custom keys used to store the resolution as MPP (micron per pixel) in WSI metadata.",
    )
    parser.add_argument(
        "--custom_list_of_wsis",
        type=str,
        default=None,
        help='Path to a CSV file specifying a custom list of WSIs to process. Must contain a "wsi" column and optionally an "mpp" column.',
    )
    parser.add_argument(
        "--reader_type",
        type=str,
        choices=["openslide", "image", "cucim"],
        default=None,
        help='Force the use of a specific WSI image reader. Options are ["openslide", "image", "cucim"]. Defaults to None (auto-determine which reader to use).',
    )
    parser.add_argument(
        "--search_nested",
        action="store_true",
        help=(
            "If set, recursively search for whole-slide images (WSIs) within all subdirectories of "
            "`wsi_dir`. Uses `os.walk` to include slides from nested folders. "
            "Defaults to False (only top-level slides are included)."
        ),
    )

    # Segmentation arguments
    parser.add_argument(
        "--segmenter",
        type=str,
        default="hest",
        choices=["hest", "grandqc", "classic", "clam"],
        help="Type of tissue vs background segmenter model to use. Options are HEST or GrandQC.",
    )
    parser.add_argument(
        "--seg_conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold to apply to binarize segmentation predictions. Lower this threshold to retain more tissue. Defaults to 0.5. Try 0.4 as 2nd option.",
    )
    parser.add_argument(
        "--remove_holes",
        action="store_true",
        default=False,
        help="If set, removes holes detected within tissue regions from the segmentation mask.",
    )
    parser.add_argument(
        "--remove_artifacts",
        action="store_true",
        default=False,
        help="If set, runs an additional GrandQC-based model to remove artifacts (including penmarks, blurs, stains, etc.) from the tissue segmentation.",
    )
    parser.add_argument(
        "--remove_penmarks",
        action="store_true",
        default=False,
        help="If set (and --remove_artifacts is not set), runs a specialized GrandQC-based model to remove only penmarks from the tissue segmentation.",
    )
    parser.add_argument(
        "--seg_batch_size",
        type=int,
        default=None,
        help="Batch size for segmentation. Defaults to None (use `batch_size` argument instead).",
    )

    # Patching arguments
    parser.add_argument(
        "--mag",
        type=int,
        choices=[5, 10, 20, 40, 80],
        default=20,
        help="Magnification level (e.g., 20 for 20x) at which to extract patches and features.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Side length of square patches in pixels at the specified magnification.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Absolute overlap between adjacent patches in pixels (at the specified magnification). Defaults to 0.",
    )
    parser.add_argument(
        "--min_tissue_proportion",
        type=float,
        default=0.0,
        help="Minimum proportion of the patch area that must contain tissue to be kept. Between 0. and 1.0. Defaults to 0. (any tissue).",
    )
    parser.add_argument(
        "--coords_dir_name",
        type=str,
        default=None,  # Changed from coords_dir
        help="Name of the directory to save/restore tissue coordinates (relative to job_dir). If None, auto-generated.",
    )

    # Feature extraction arguments
    parser.add_argument(
        "--patch_encoder",
        type=str,
        default="conch_v15",
        choices=[  # List all supported patch encoders from aegis.feature_extraction.models.patch_encoders.factory
            "conch_v1",
            "uni_v1",
            "uni_v2",
            "ctranspath",
            "phikon",
            "resnet50",
            "gigapath",
            "virchow",
            "virchow2",
            "hoptimus0",
            "hoptimus1",
            "phikon_v2",
            "conch_v15",
            "musk",
            "hibou_l",
            "kaiko-vits8",
            "kaiko-vits16",
            "kaiko-vitb8",
            "kaiko-vitb16",
            "kaiko-vitl14",
            "lunit-vits8",
            "midnight12k",
        ],
        help="Patch encoder model to use for feature extraction.",
    )
    parser.add_argument(
        "--patch_encoder_ckpt_path",
        type=str,
        default=None,
        help=(
            "Optional local path to a patch encoder checkpoint (.pt, .pth, .bin, or .safetensors). "
            "This overrides the default download mechanism and model registry. "
            "Useful for offline environments or custom checkpoints."
        ),
    )
    parser.add_argument(
        "--slide_encoder",
        type=str,
        default=None,
        choices=[  # List all supported slide encoders from aegis.feature_extraction.models.slide_encoders.factory
            "threads",
            "titan",
            "prism",
            "gigapath",
            "chief",
            "madeleine",
            # Mean-pooling variants (derived from patch encoders)
            "mean-virchow",
            "mean-virchow2",
            "mean-conch_v1",
            "mean-conch_v15",
            "mean-ctranspath",
            "mean-gigapath",
            "mean-resnet50",
            "mean-hoptimus0",
            "mean-phikon",
            "mean-phikon_v2",
            "mean-musk",
            "mean-uni_v1",
            "mean-uni_v2",
            "mean-hibou_l",
            "mean-lunit-vits8",
            "mean-midnight12k",
            "mean-kaiko-vits8",
            "mean-kaiko-vits16",
            "mean-kaiko-vitb8",
            "kaiko-vitb16",
            "kaiko-vitl14",
        ],
        help="Slide encoder model to use for feature extraction. If specified, will automatically extract required patch features.",
    )
    parser.add_argument(
        "--feat_batch_size",
        type=int,
        default=None,
        help="Batch size for feature extraction. Defaults to None (use `batch_size` argument instead).",
    )

    return parser


def get_mil_args(parser: argparse.ArgumentParser):
    # Data & I/O Configuration
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default=None,
        help="Specify the root directory where the dataset is located. This is essential for loading the data correctly.",
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default=None,
        help="Path to the CSV file with slide ids, patient ids, and labels.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="Path to the validation CSV file.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to the testing CSV file.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Name of the column containing the labels in the dataset CSV.",
    )
    parser.add_argument(
        "--patient_id_col",
        type=str,
        default="case_id",
        help="Name of the column containing patient IDs.",
    )
    parser.add_argument(
        "--slide_id_col",
        type=str,
        default="slide_id",
        help="Name of the column containing slide IDs.",
    )
    parser.add_argument(
        "--metadata_cols",
        type=str,
        default=None,
        help="Comma-separated CSV column names to use as extra modalities (e.g. OncoTreeSiteCode). Encoded as one-hot and concatenated with patch features.",
    )
    parser.add_argument(
        "--results_dir",
        default="./results",
        help="Path to the directory where training results and model checkpoints will be saved. Default is './results'.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Path to the directory containing custom data splits. If not provided, splits will be generated based on the task and label fraction.",
    )
    parser.add_argument(
        "--patch_size",
        type=str,
        default="",
        help="Define the size of image patches in the format [height]x[width]. This is important for processing images.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="20x",
        help="Set the magnification level for processing images. Examples include '10x' or '10x_40x' for combined levels.",
    )
    parser.add_argument(
        "--early_fusion",
        action="store_true",  # Use action='store_true' for boolean flags with default False
        help="Enable or disable early fusion for models that utilize multiple magnification levels. This can enhance model performance.",
    )
    parser.add_argument(
        "--preloading",
        choices=["yes", "no"],
        default="no",
        help="Specify whether to preload data into memory for faster access during training. Options are 'yes' or 'no'.",
    )
    parser.add_argument(
        "--use_hdf5",
        action="store_true",
        help="Enable the use of HDF5 files for feature storage.",
    )
    parser.add_argument(
        "--memmap_bin_path",
        type=str,
        default=None,
        help="Path to the binary memmap file containing concatenated features. If provided along with --memmap_json_path, will use memory-mapped datasets for faster I/O.",
    )
    parser.add_argument(
        "--memmap_json_path",
        type=str,
        default=None,
        help="Path to the JSON index file mapping slide_ids to [start_row, num_rows] in the memmap binary file. Required if --memmap_bin_path is provided.",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Set the maximum number of epochs for training the model. Default is 200.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for the optimizer. Adjust this for better convergence.",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-5,
        help="Weight decay factor for L2 regularization. Helps prevent overfitting.",
    )
    parser.add_argument(
        "--opt",
        choices=["adam", "sgd", "adamw"],
        default="adam",
        help="Choose the optimizer to use for training. Options include 'adam', 'sgd', or 'adamw'.",
    )
    parser.add_argument(
        "--drop_out",
        type=float,
        default=0.25,
        help="Set the dropout probability to prevent overfitting during training.",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping to halt training when validation performance stops improving.",
    )
    parser.add_argument(
        "--weighted_sample",
        action="store_true",
        help="Enable weighted sampling to address class imbalance in the training dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Set the batch size for training.",
    )

    # Model Configuration
    parser.add_argument(
        "--model_type",
        type=str,
        default="att_mil",
        help="Specify the type of model architecture to use for training. Default is 'att_mil'.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Select the backbone network for feature extraction. Default is 'resnet50'.",
    )
    parser.add_argument(
        "--in_dim",
        type=int,
        default=1024,
        help="Set the input dimension for the model. This should match the output of the backbone network.",
    )

    # MambaMIL Specific Configuration
    parser.add_argument(
        "--mambamil_rate",
        type=int,
        default=10,
        help="Rate parameter for MambaMIL, influencing the model's behavior.",
    )
    parser.add_argument(
        "--mambamil_layer",
        type=int,
        default=2,
        help="Number of layers in the MambaMIL architecture.",
    )
    parser.add_argument(
        "--mambamil_type",
        choices=["Mamba", "BiMamba", "SRMamba"],
        default="SRMamba",
        help="Select the type of Mamba architecture to use. Options include 'Mamba', 'BiMamba', or 'SRMamba'.",
    )

    # Experiment & Reproducibility
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Specify the task name or identifier for the experiment.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        help="Specify the task type ('classification' or 'survival').",
    )
    parser.add_argument(
        "--exp_code",
        type=str,
        required=True,
        help="Provide a unique experiment code for tracking purposes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Set the random seed for reproducibility of results. Default is 1.",
    )
    parser.add_argument(
        "--label_frac",
        type=float,
        default=1.0,
        help="Specify the fraction of training labels to use. Default is 1.0 (use all labels).",
    )
    parser.add_argument(
        "--log_data",
        action="store_true",
        help="Enable logging of training data using TensorBoard for visualization and analysis.",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Enable testing/debugging mode for the experiment.",
    )

    # Cross-Validation Configuration
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Specify the total number of folds for cross-validation. Default is 10.",
    )
    parser.add_argument(
        "--k_start",
        type=int,
        default=-1,
        help="Set the starting fold for cross-validation. Use -1 for the last fold.",
    )
    parser.add_argument(
        "--k_end",
        type=int,
        default=-1,
        help="Set the ending fold for cross-validation. Use -1 for the first fold.",
    )

    # Survival Configuration
    parser.add_argument(
        "--bag_loss",
        type=str,
        choices=["svm", "ce", "ce_surv", "nll_surv", "cox_surv"],
        default="nll_surv",
        help="Slide-level classification loss function (default: nll_surv).",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["focal", "poly"],
        default="focal",
        help="Type of loss function to use for classification tasks. Options: 'focal' (default), 'poly'.",
    )
    parser.add_argument(
        "--alpha_surv",
        type=float,
        default=0.0,
        help="How much to weigh uncensored patients.",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=1e-4,
        help="L1-Regularization Strength (Default 1e-4).",
    )
    parser.add_argument(
        "--inst_loss",
        type=str,
        choices=["svm", "ce", None],
        default=None,
        help="Instance-level clustering loss function (default: None).",
    )
    parser.add_argument(
        "--subtyping",
        action="store_true",  # Use action='store_true' for boolean flags with default False
        help="Enable subtyping problem.",
    )
    parser.add_argument(
        "--bag_weight",
        type=float,
        default=0.7,
        help="Weight coefficient for bag-level loss (default: 0.7).",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=8,
        help="Number of positive/negative patches to sample for clam.",
    )
    parser.add_argument(
        "--gc",
        type=int,
        default=32,
        help="Gradient Accumulation Step.",
    )
    parser.add_argument(
        "--n_subsamples",
        type=int,
        default=-1,
        help="Number of patches to sample per bag during training. -1 means use all patches. This is essential for MIL training with large bags.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of subprocesses to use for data loading. Default is 16 for faster data loading.",
    )
