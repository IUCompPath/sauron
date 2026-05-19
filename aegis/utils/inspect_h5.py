import argparse
import glob
import os

import h5py
import numpy as np
from tqdm import tqdm


def compare_datasets(combined_dset, original_dset, path=""):
    """
    Compare two HDF5 datasets and return any discrepancies.

    Returns:
        list of error messages (empty if no errors)
    """
    errors = []

    # Compare shapes
    if combined_dset.shape != original_dset.shape:
        errors.append(
            f"  Shape mismatch at {path}: combined={combined_dset.shape}, original={original_dset.shape}"
        )

    # Compare dtypes
    if combined_dset.dtype != original_dset.dtype:
        errors.append(
            f"  Dtype mismatch at {path}: combined={combined_dset.dtype}, original={original_dset.dtype}"
        )

    # Compare values (only if shapes match)
    if combined_dset.shape == original_dset.shape:
        try:
            combined_data = combined_dset[...]
            original_data = original_dset[...]

            if not np.array_equal(combined_data, original_data):
                # Check if it's a floating point precision issue
                if np.issubdtype(combined_dset.dtype, np.floating):
                    if not np.allclose(
                        combined_data, original_data, rtol=1e-5, atol=1e-8
                    ):
                        max_diff = np.max(np.abs(combined_data - original_data))
                        errors.append(
                            f"  Value mismatch at {path}: max difference = {max_diff}"
                        )
                else:
                    errors.append(f"  Value mismatch at {path}: arrays are not equal")
        except Exception as e:
            errors.append(f"  Error reading values at {path}: {e}")

    return errors


def compare_h5_structures(combined_obj, original_obj, path="", errors=None):
    """
    Recursively compare HDF5 structures between combined and original files.

    Args:
        combined_obj: Object from combined H5 file
        original_obj: Object from original H5 file
        path: Current path in hierarchy
        errors: List to accumulate errors
    """
    if errors is None:
        errors = []

    if isinstance(combined_obj, h5py.Dataset) and isinstance(
        original_obj, h5py.Dataset
    ):
        # Both are datasets - compare them
        dataset_errors = compare_datasets(combined_obj, original_obj, path)
        errors.extend(dataset_errors)

    elif isinstance(combined_obj, h5py.Group) and isinstance(original_obj, h5py.Group):
        # Both are groups - compare their contents
        combined_keys = set(combined_obj.keys())
        original_keys = set(original_obj.keys())

        # Check for missing keys in combined
        missing_in_combined = original_keys - combined_keys
        if missing_in_combined:
            errors.append(
                f"  Missing keys in combined file at {path}: {missing_in_combined}"
            )

        # Check for extra keys in combined
        extra_in_combined = combined_keys - original_keys
        if extra_in_combined:
            errors.append(
                f"  Extra keys in combined file at {path}: {extra_in_combined}"
            )

        # Compare common keys
        common_keys = combined_keys & original_keys
        for key in common_keys:
            new_path = f"{path}/{key}" if path else key
            compare_h5_structures(
                combined_obj[key], original_obj[key], new_path, errors
            )

    else:
        # Type mismatch
        errors.append(
            f"  Type mismatch at {path}: combined={type(combined_obj).__name__}, "
            f"original={type(original_obj).__name__}"
        )

    return errors


def verify_combined_h5(source_dir, combined_file):
    """
    Verify that the combined H5 file matches all original individual H5 files.

    Args:
        source_dir: Directory containing original .h5 files
        combined_file: Path to the combined H5 file
    """
    if not os.path.exists(combined_file):
        print(f"Error: Combined file not found: {combined_file}")
        return False

    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return False

    print(f"Verifying combined file: {combined_file}")
    print(f"Against source directory: {source_dir}\n")
    print("=" * 80)

    # Create a mapping of base names to file paths
    source_files = glob.glob(os.path.join(source_dir, "*.h5"))
    base_name_to_path = {
        os.path.splitext(os.path.basename(f))[0]: f for f in source_files
    }

    total_errors = 0
    files_verified = 0
    files_failed = 0
    files_missing = 0

    try:
        with h5py.File(combined_file, "r") as combined_h5:
            # Get all top-level groups (each corresponds to a source file)
            combined_groups = list(combined_h5.keys())

            print(f"Found {len(combined_groups)} groups in combined file")
            print(f"Found {len(source_files)} source files\n")

            # Verify each group in combined file
            for base_name in tqdm(combined_groups, desc="Verifying files"):
                if base_name not in base_name_to_path:
                    print(f"\n⚠️  WARNING: No source file found for group '{base_name}'")
                    files_missing += 1
                    continue

                source_file = base_name_to_path[base_name]
                file_name = os.path.basename(source_file)

                try:
                    with h5py.File(source_file, "r") as original_h5:
                        # Compare the group in combined file with the entire original file
                        combined_group = combined_h5[base_name]

                        errors = compare_h5_structures(
                            combined_group, original_h5, base_name
                        )

                        if errors:
                            print(f"\n❌ FAILED: {file_name}")
                            for error in errors:
                                print(error)
                            total_errors += len(errors)
                            files_failed += 1
                        else:
                            files_verified += 1
                            # Print success for first few files, then suppress
                            if files_verified <= 5:
                                print(f"\n✅ VERIFIED: {file_name}")

                except OSError as e:
                    print(f"\n❌ ERROR: Cannot open source file {file_name}: {e}")
                    files_failed += 1
                except Exception as e:
                    print(f"\n❌ ERROR: Error processing {file_name}: {e}")
                    files_failed += 1

            # Check for source files that weren't in combined file
            combined_base_names = set(combined_groups)
            source_base_names = set(base_name_to_path.keys())
            missing_in_combined = source_base_names - combined_base_names

            if missing_in_combined:
                print(
                    f"\n⚠️  WARNING: {len(missing_in_combined)} source files not found in combined file:"
                )
                for base_name in sorted(missing_in_combined)[:10]:  # Show first 10
                    print(f"    - {base_name}.h5")
                if len(missing_in_combined) > 10:
                    print(f"    ... and {len(missing_in_combined) - 10} more")

    except OSError as e:
        print(f"Error: Cannot open combined file - file may be corrupted: {e}")
        return False
    except Exception as e:
        print(f"Error verifying file: {e}")
        return False

    # Print summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"✅ Files verified successfully: {files_verified}")
    print(f"❌ Files with errors: {files_failed}")
    print(f"⚠️  Source files missing from combined: {files_missing}")
    if missing_in_combined:
        print(f"⚠️  Combined file missing {len(missing_in_combined)} source files")
    print(f"📊 Total errors found: {total_errors}")

    if total_errors == 0 and files_failed == 0 and files_missing == 0:
        print("\n🎉 All files verified successfully! Combined file matches originals.")
        return True
    else:
        print("\n⚠️  Verification found discrepancies. Please review the errors above.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify combined H5 file against original individual H5 files."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="E:\\features_uni_v2",
        help="Directory containing original .h5 files",
    )
    parser.add_argument(
        "--combined_file",
        type=str,
        default="E:\\combined_features_univ2_20x_256.h5",
        help="Path to the combined H5 file to verify",
    )

    args = parser.parse_args()

    verify_combined_h5(args.source_dir, args.combined_file)
