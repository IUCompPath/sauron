import argparse


def main(args):
    """Main function for training the model."""
    # Placeholder for actual training logic
    print(f"Starting training with the following configurations:\n{args}")


def get_args():
    """
    Parse command line arguments for Whole Slide Image (WSI) Training configurations.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Configurations for Whole Slide Image (WSI) Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset and paths
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to save training results and model checkpoints",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help=(
            "Directory containing custom data splits. If not specified, splits will be "
            "inferred from the task and label_frac arguments"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--csv_fpath",
        type=str,
        default=None,
        help="Path to CSV file containing labels",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Cohort or disease model identifier",
    )

    # Training hyperparameters
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for the optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization) factor",
    )
    parser.add_argument(
        "--opt",
        type=str,
        choices=["adam", "adamW", "sgd"],
        default="adamW",
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--drop_out",
        type=float,
        default=0.25,
        help="Dropout probability for regularization",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping to prevent overfitting",
    )
    parser.add_argument(
        "--weighted_sample",
        action="store_true",
        help="Enable weighted sampling to handle class imbalance",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--n_subsamples",
        type=int,
        default=-1,
        help="Number of patches to sample during training",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="Scheduler used for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of CPU workers for data loading",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.001,
        help="Temperature parameter for training",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=False,
        help="Enable warmup",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of epochs for warmup",
    )
    parser.add_argument(
        "--end_learning_rate",
        type=float,
        default=1e-8,
        help="End learning rate for scheduler",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        help="Precision for training (e.g., float32, float64)",
    )

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        default="att_mil",
        choices=[
            "att_mil",
            "trans_mil",
            "diff_att_mil",
            "mean_mil",
            "max_mil",
            "Mamba",
            "BiMamba",
            "SRMamba",
        ],
        help="Type of model architecture (e.g., 'att_mil')",
    )
    parser.add_argument(
        "--wsi_encoder",
        type=str,
        default="abmil",
        help="WSI encoder to use",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Backbone network for feature extraction",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="softmax",
        help="Activation function to use",
    )
    parser.add_argument(
        "--in_dim",
        type=int,
        default=1024,
        help="Input dimension for the model",
    )
    parser.add_argument(
        "--wsi_encoder_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for WSI encoder",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Number of heads in attention mechanism",
    )
    parser.add_argument(
        "--add_stain_encoding",
        action="store_true",
        default=False,
        help="Include stain encodings in the model",
    )
    parser.add_argument(
        "--mambamil_rate",
        type=int,
        default=10,
        help="Rate parameter for MambaMIL",
    )
    parser.add_argument(
        "--mambamil_layer",
        type=int,
        default=2,
        help="Number of layers in MambaMIL",
    )
    # Experiment settings
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name or identifier for the current experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--label_frac",
        type=float,
        default=1.0,
        help="Fraction of training labels to use",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Total number of folds for cross-validation",
    )
    parser.add_argument(
        "--k_start",
        type=int,
        default=-1,
        help="Starting fold for cross-validation (-1 for last fold)",
    )
    parser.add_argument(
        "--k_end",
        type=int,
        default=-1,
        help="Ending fold for cross-validation (-1 for first fold)",
    )

    # Data processing
    parser.add_argument(
        "--patch_size",
        type=str,
        default="",
        help="Size of image patches (format: [height]x[width])",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="20x",
        help=(
            "Magnification level to work with, e.g., '10x' or '10x_40x' "
            "(for multiple levels)"
        ),
    )
    parser.add_argument(
        "--early_fusion",
        action="store_true",
        default=False,
        help=(
            "Create an early fusion instead of a late fusion of models with "
            "multiple magnification levels"
        ),
    )
    parser.add_argument(
        "--preloading",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="Whether to preload data into memory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading",
    )

    # Logging and debugging
    parser.add_argument(
        "--log_data",
        action="store_true",
        help="Enable logging of training data using TensorBoard",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Enable testing/debugging mode",
    )
    parser.add_argument(
        "--log_ml",
        action="store_true",
        help="Enable logging of results in MLflow and TensorBoard",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="WSI_Project",
        help="Project name to use for logging to WANDB",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="wsi_entity",
        help="Entity to use for logging to WANDB",
    )

    # Loss functions
    parser.add_argument(
        "--symmetric_cl",
        action="store_true",
        default=False,
        help="Use symmetric contrastive loss",
    )
    parser.add_argument(
        "--global_loss",
        type=str,
        default="-1",
        help="Loss used for global alignment of different WSI",
    )
    parser.add_argument(
        "--local_loss",
        type=str,
        default="-1",
        help="Loss used for local alignment of different WSI",
    )
    parser.add_argument(
        "--intra_modality_loss",
        type=str,
        default="-1",
        help="Info-NCE loss for comparing different views of the same WSI",
    )
    parser.add_argument(
        "--local_loss_weight",
        type=float,
        default=1.0,
        help="Weight for local loss",
    )

    # Pretrained model
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to directory with checkpoint",
    )

    args = parser.parse_args()
    return args
