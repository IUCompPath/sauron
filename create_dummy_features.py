import argparse
import os
import pandas as pd
import numpy as np
import h5py


def create_dummy_features(csv_path, output_dir="dummy_features"):
    """
    Reads a CSV file with slide IDs and creates dummy HDF5 feature files.

    Args:
        csv_path (str): Path to the input CSV file.
        output_dir (str): Directory to save the HDF5 files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    if "slide_id" not in df.columns:
        print("Error: 'slide_id' column not found in the CSV file.")
        return

    for slide_id in df["slide_id"]:
        # Generate a random number of features (N) between 200 and 2001
        n_features = np.random.randint(200, 2001)

        # Create dummy features of size (N, 1024)
        features = np.random.rand(n_features, 1024).astype(np.float32)

        # Define the output HDF5 file path
        h5_filename = f"{slide_id}.h5"
        h5_filepath = os.path.join(output_dir, h5_filename)

        try:
            # Store the features in an HDF5 file
            with h5py.File(h5_filepath, "w") as hf:
                hf.create_dataset("features", data=features)

            print(f"Created {h5_filepath} with features of shape {features.shape}")

        except Exception as e:
            print(f"An error occurred while creating {h5_filepath}: {e}")


def main():
    """
    Main function to parse arguments and run the feature creation process.
    """
    parser = argparse.ArgumentParser(
        description="Create dummy HDF5 feature files from a CSV of slide IDs."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the input CSV file containing slide IDs.",
    )
    args = parser.parse_args()

    create_dummy_features(args.csv_file)


if __name__ == "__main__":
    main()
