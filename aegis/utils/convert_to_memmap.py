import argparse
import glob
import json
import os

import h5py
import numpy as np
from tqdm import tqdm


def convert_to_raw_memmap(source_dir, output_bin, output_json):
    """
    Reads individual .h5 files, extracts the 'features' dataset,
    concatenates them into a raw binary file, and saves an index JSON.

    Result:
    1. output_bin: A massive float32 byte stream.
    2. output_json: { "slide_id": [start_row_index, num_rows], ... }
    """

    # 1. Find files
    h5_files = glob.glob(os.path.join(source_dir, "*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {source_dir}")
        return

    # Sort files to ensure deterministic order (optional but recommended)
    h5_files.sort()

    print(f"Found {len(h5_files)} files. Converting to Raw Binary...")

    # Metadata storage
    slide_index = {}
    global_row_counter = 0
    total_expected_floats = 0

    # The fixed column size (dimension)
    FEATURE_DIM = 1536

    # 2. Open Binary Output File
    with open(output_bin, "wb") as f_out:
        for file_path in tqdm(h5_files, desc="Flattening"):
            slide_id = os.path.splitext(os.path.basename(file_path))[0]

            try:
                with h5py.File(file_path, "r") as src_h5:
                    # Check if 'features' key exists
                    if "features" not in src_h5:
                        print(f"Skipping {slide_id}: 'features' key missing.")
                        continue

                    # Read features into RAM
                    # shape: [N, 1536]
                    data = src_h5["features"][:]

                    # Safety Check: dimensions
                    if data.ndim != 2 or data.shape[1] != FEATURE_DIM:
                        print(
                            f"Warning: {slide_id} has weird shape {data.shape}. Skipping."
                        )
                        continue

                    # Safety Check: Ensure float32 (4 bytes)
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)

                    # 3. Write Raw Bytes
                    # .tobytes() creates a C-contiguous byte string
                    f_out.write(data.tobytes())

                    # 4. Record Index
                    # We store [start_index, num_rows] relative to the combined file
                    num_rows = data.shape[0]
                    slide_index[slide_id] = [global_row_counter, num_rows]

                    global_row_counter += num_rows

            except OSError:
                print(f"Error: Corrupted file {file_path}")
            except Exception as e:
                print(f"Error processing {slide_id}: {e}")

    # 3. Save Index JSON
    meta_data = {
        "feature_dim": FEATURE_DIM,
        "dtype": "float32",
        "total_rows": global_row_counter,
        "slides": slide_index,
    }

    with open(output_json, "w") as f_json:
        json.dump(meta_data, f_json, indent=4)

    # 4. Verification Statistics
    print("\n--- Conversion Complete ---")
    print(f"Total Rows: {global_row_counter:,}")
    print(f"Feature Dim: {FEATURE_DIM}")

    expected_size_bytes = global_row_counter * FEATURE_DIM * 4  # 4 bytes for float32
    actual_size_bytes = os.path.getsize(output_bin)

    print(f"Expected Size: {expected_size_bytes / (1024**3):.2f} GB")
    print(f"Actual Size:   {actual_size_bytes / (1024**3):.2f} GB")

    if expected_size_bytes == actual_size_bytes:
        print("SUCCESS: File integrity verified.")
    else:
        print("WARNING: Byte size mismatch! Something went wrong.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir", type=str, required=True, help="Folder containing .h5 files"
    )
    parser.add_argument(
        "--output_bin", type=str, default="dataset.bin", help="Output raw binary file"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="dataset_index.json",
        help="Output index file",
    )

    args = parser.parse_args()

    convert_to_raw_memmap(args.source_dir, args.output_bin, args.output_json)
