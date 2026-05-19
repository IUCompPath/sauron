import h5py

# Replace with the actual path to one of the failing files
bad_file = "/mnt/e/features_uni_v2/TCGA-ZT-A8OM-01Z-00-DX1.4844377F-3082-4CEE-9B1E-FC17EA06D150.h5"

try:
    with h5py.File(bad_file, "r") as f:
        print("Keys in file:", list(f.keys()))
        print("File opened successfully.")
except Exception as e:
    print(f"CRITICAL FAILURE: {e}")
