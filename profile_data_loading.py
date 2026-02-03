import argparse
import time
from typing import Dict

from tqdm import tqdm

from aegis.data.data_utils import get_dataloader
from aegis.data.dataset_factory import get_data_manager
from aegis.parse.cli_parsers import get_mil_args


def profile_dataloader_iteration(
    dataset,
    num_workers: int,
    batch_size: int,
    use_hdf5: bool,
    n_subsamples: int,
    task_type: str,
    num_epochs: int = 2,
) -> float:
    """Times iterating through a DataLoader for a few epochs."""
    print(
        f"--- Profiling DataLoader: workers={num_workers}, hdf5={use_hdf5}, batch_size={batch_size} ---"
    )

    if not hasattr(dataset, "load_from_hdf5"):
        print("Warning: Dataset does not support toggling HDF5 mode.")
        return -1.0
    dataset.load_from_hdf5(use_hdf5)

    # Disable caching for this test to measure pure I/O
    if hasattr(dataset, "cache_enabled"):
        dataset.cache_enabled = False

    try:
        loader = get_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn_type=task_type,
            n_subsamples=n_subsamples,
            num_workers=num_workers,
        )
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        return -1.0

    total_time = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        try:
            for _ in tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{num_epochs} (workers={num_workers}, hdf5={use_hdf5})",
            ):
                # In a real scenario, this is where data is moved to the device.
                # We are only interested in the loading time.
                pass
        except FileNotFoundError as e:
            print(f"\nError during iteration: {e}")
            print("Please ensure your --data_root_dir is correct and features exist.")
            return -1.0
        except Exception as e:
            print(f"\nAn unexpected error occurred during iteration: {e}")
            return -1.0

        epoch_time = time.time() - start_time
        total_time += epoch_time
        print(f"Epoch {epoch+1} time: {epoch_time:.4f} seconds")

    avg_time = total_time / num_epochs
    print(f"Average iteration time over {num_epochs} epochs: {avg_time:.4f} seconds\n")
    return avg_time


def profile_preloading(dataset, use_hdf5: bool) -> float:
    """Times the dataset's preload_data() method."""
    print(f"--- Profiling Preloading: hdf5={use_hdf5} ---")

    if not hasattr(dataset, "preload_data"):
        print("Warning: Dataset does not have a preload_data method.")
        return -1.0
    if not hasattr(dataset, "load_from_hdf5"):
        print("Warning: Dataset does not support toggling HDF5 mode.")
        return -1.0

    dataset.load_from_hdf5(use_hdf5)

    start_time = time.time()
    try:
        dataset.preload_data()
    except FileNotFoundError as e:
        print(f"\nError during preloading: {e}")
        print("Please ensure your --data_root_dir is correct and features exist.")
        return -1.0
    except Exception as e:
        print(f"\nAn unexpected error occurred during preloading: {e}")
        return -1.0

    preload_time = time.time() - start_time
    print(f"Preloading time: {preload_time:.4f} seconds\n")
    return preload_time


def main():
    """Main function to run the data loading profiling."""
    parser = argparse.ArgumentParser(
        description="Aegis Data Loading Profiler",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    get_mil_args(parser)
    parser.add_argument(
        "--profile_num_workers",
        nargs="+",
        type=int,
        default=[0, 4, 8, 16],
        help="List of num_workers values to test for DataLoader iteration.",
    )
    parser.add_argument(
        "--profile_epochs",
        type=int,
        default=2,
        help="Number of epochs to iterate through for timing analysis.",
    )
    parser.add_argument(
        "--profile_batch_size",
        type=int,
        default=None,
        help="Override batch_size for profiling. If not set, uses the value from --batch_size.",
    )

    args = parser.parse_args()

    # Override batch_size if the profiler-specific one is provided
    if args.profile_batch_size:
        args.batch_size = args.profile_batch_size

    print("--- Initializing DataManager (verbose output is off) ---")
    # This setup is simplified from the main training script
    manager_params = {
        "task_name": getattr(args, "task_name", args.task),
        "task_type": args.task_type,
        "csv_path": args.dataset_csv,
        "data_directory": args.data_root_dir,
        "seed": args.seed,
        "verbose": False,
        "label_column": args.label_col,
        "patient_id_col_name": args.patient_id_col,
        "slide_id_col_name": args.slide_id_col,
        "split_dir": getattr(args, "split_dir", None),
    }
    data_manager = get_data_manager(**manager_params)
    data_manager.create_k_fold_splits(num_folds=args.k, test_set_size=0.1)
    data_manager.set_current_fold(fold_index=args.k_start)

    mil_dataset_params = {
        "backbone": args.backbone,
        "patch_size": args.patch_size,
        "use_hdf5": True,  # Will be toggled during tests
        "cache_enabled": False,  # Manually controlled during tests
        "n_subsamples": args.n_subsamples,
    }

    train_dataset, _, _ = data_manager.get_mil_datasets(**mil_dataset_params)

    if not train_dataset:
        print("\nCould not create the training dataset. Exiting.")
        return

    print(f"Successfully created dataset with {len(train_dataset)} samples.")
    print("--- Starting Profiling ---\n")

    results: Dict[str, Dict] = {"iteration_hdf5": {}, "iteration_pt": {}}

    # --- Test 1: Preloading ---
    results["preload_hdf5_time"] = profile_preloading(train_dataset, use_hdf5=True)
    results["preload_pt_time"] = profile_preloading(train_dataset, use_hdf5=False)

    # --- Test 2: DataLoader Iteration (HDF5 vs PT for each num_workers) ---
    for workers in sorted(list(set(args.profile_num_workers))):
        # Test with HDF5
        results["iteration_hdf5"][workers] = profile_dataloader_iteration(
            train_dataset,
            workers,
            args.batch_size,
            use_hdf5=True,
            n_subsamples=args.n_subsamples,
            task_type=args.task_type,
            num_epochs=args.profile_epochs,
        )
        # Test with .pt files
        results["iteration_pt"][workers] = profile_dataloader_iteration(
            train_dataset,
            workers,
            args.batch_size,
            use_hdf5=False,
            n_subsamples=args.n_subsamples,
            task_type=args.task_type,
            num_epochs=args.profile_epochs,
        )

    # --- Print Summary Report ---
    print("\n\n--- Profiling Summary ---")
    print("\n* Preloading Times *")
    print(
        f"  - HDF5 (.h5) files:    {results.get('preload_hdf5_time', -1):.4f} seconds"
    )
    print(f"  - PyTorch (.pt) files: {results.get('preload_pt_time', -1):.4f} seconds")
    print("\n* DataLoader Iteration Times (Average per Epoch) *")

    header = f"| {'Num Workers':<12} | {'HDF5 (.h5)':<15} | {'PyTorch (.pt)':<15} |"
    print(header)
    print("-" * len(header))
    for w in sorted(results["iteration_hdf5"].keys()):
        hdf5_time_val = results["iteration_hdf5"].get(w)
        pt_time_val = results["iteration_pt"].get(w)

        hdf5_time_str = f"{hdf5_time_val:.4f}s" if hdf5_time_val != -1 else "FAIL"
        pt_time_str = f"{pt_time_val:.4f}s" if pt_time_val != -1 else "FAIL"

        row = f"| {w:<12} | {hdf5_time_str:<15} | {pt_time_str:<15} |"
        print(row)

    print("\n--- How to Interpret Results ---")
    print(
        "1. Preloading Times: This is the initial, one-time cost if you use `--preloading yes`. A high value explains a long startup delay."
    )
    print(
        "2. DataLoader Iteration Times: This is the per-epoch cost of loading data if you do NOT preload. Lower is better."
    )
    print(
        "   - For each column (HDF5 vs .pt), find the `Num Workers` with the lowest time. This is the optimal setting for that file type."
    )
    print(
        "   - Compare the best HDF5 time with the best .pt time to see which file format is faster for your system."
    )
    print(
        "\nDecision: If the best 'DataLoader Iteration Time' is lower than the 'Preloading Time', you should probably avoid preloading. If preloading is faster and the initial delay is acceptable, it might be a good choice."
    )
    print(
        "\nNote: `tqdm` is used for progress bars. If not installed, run: `pip install tqdm`"
    )


if __name__ == "__main__":
    main()
