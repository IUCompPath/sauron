# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Step 1: Load Your Data ---
# NOTE: Replace 'your_file.csv' with the actual path to your CSV file.
try:
    # This is the line you should use to load your own file
    df = pd.read_csv(
        r"C:\Users\thakursp\Documents\Work\sauron\aegis\utils\wsi-detailed-report.csv"
    )

    # For demonstration purposes, we will simulate the 11,000-row dataset
    # # based on the structure you provided.
    # print("Loading and simulating data for demonstration...")
    # num_rows = 11000
    # subtypes = ["TCGA-PRAD", "TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC", "TCGA-LGG"]
    # vendors = ["aperio", "hamamatsu", "leica"]
    # objective_powers = [20, 40]

    # simulated_data = {
    #     "subtype": np.random.choice(subtypes, num_rows, p=[0.1, 0.3, 0.25, 0.15, 0.2]),
    #     "vendor": np.random.choice(vendors, num_rows, p=[0.6, 0.3, 0.1]),
    #     "objective_power": np.random.choice(objective_powers, num_rows, p=[0.3, 0.7]),
    #     "level_count": np.random.randint(3, 9, size=num_rows),
    #     "level_0_width": np.random.randint(20000, 120000, size=num_rows)
    #     + np.random.randn(num_rows) * 5000,
    #     "level_0_height": np.random.randint(20000, 120000, size=num_rows)
    #     + np.random.randn(num_rows) * 5000,
    # }
    # df = pd.DataFrame(simulated_data)

    # # We introduce some missing values to demonstrate NAN handling
    # df.loc[df.sample(frac=0.03, random_state=1).index, "level_0_width"] = np.nan
    # print("Data simulation complete.")

except FileNotFoundError:
    print("Error: The file 'your_file.csv' was not found.")
    print(
        "Please make sure the CSV file is in the same directory as the script or provide the full path."
    )
    exit()


# --- Step 2: Basic Data Cleaning and Preparation ---
# Ensure key numerical columns are of a numeric type.
# 'coerce' will turn any non-numeric values into NaN (Not a Number).
print("\n--- Cleaning and Preparing Data ---")
df["level_0_width"] = pd.to_numeric(df["level_0_width"], errors="coerce")
df["level_0_height"] = pd.to_numeric(df["level_0_height"], errors="coerce")

# Create a new feature for the total number of pixels (slide area)
df["level_0_total_pixels"] = df["level_0_width"] * df["level_0_height"]

# Display data types and check for missing values
print("Data Types:\n", df[["subtype", "level_0_total_pixels", "level_count"]].dtypes)
print(
    "\nMissing Values Count:\n", df[["subtype", "level_0_total_pixels"]].isnull().sum()
)


# --- Step 3: Generate and Display Interesting Graphs ---
print("\n--- Generating Graphs ---")

# == Graph 1: Number of Slides per Cancer Subtype ==
# This graph is essential for understanding the composition of your dataset.
plt.figure(figsize=(12, 7))
sns.countplot(
    y="subtype", data=df, order=df["subtype"].value_counts().index, palette="viridis"
)
plt.title("Graph 1: Number of Slides per Cancer Subtype", fontsize=16)
plt.xlabel("Number of Slides", fontsize=12)
plt.ylabel("Cancer Subtype", fontsize=12)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# == Graph 2 (REVISED): Slide Area Distribution by Cancer Subtype ==
# This boxplot explores the distribution of slide sizes (in total pixels) for each
# cancer subtype. It can reveal if certain cancer types typically have larger
# or smaller tissue sections. We handle NANs by dropping them for this plot.
plot_data_2 = df.dropna(subset=["level_0_total_pixels", "subtype"])

# Order subtypes by their median size for better visualization
subtype_order = (
    plot_data_2.groupby("subtype")["level_0_total_pixels"].median().sort_values().index
)

plt.figure(figsize=(12, 8))
sns.boxplot(
    x="subtype",
    y="level_0_total_pixels",
    data=plot_data_2,
    order=subtype_order,
    palette="plasma",
)
plt.title("Graph 2: Slide Area Distribution by Cancer Subtype", fontsize=16)
plt.xlabel("Cancer Subtype", fontsize=12)
plt.ylabel("Total Slide Area (in billion pixels)", fontsize=12)
plt.ticklabel_format(
    style="sci", axis="y", scilimits=(9, 9)
)  # Use scientific notation for y-axis
plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# == Graph 3: Slide Size vs. Number of Magnification Levels ==
# This scatter plot explores the relationship between the physical size of the slide
# and the number of zoom levels available.
# For clarity, we plot a random sample of 2000 points to prevent overplotting.
plot_data_3 = df.dropna(subset=["level_0_total_pixels", "level_count"])
sample_size = min(2000, len(plot_data_3))  # Ensure sample size is not larger than data
plot_data_3_sample = plot_data_3.sample(n=sample_size, random_state=42)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x="level_0_total_pixels",
    y="level_count",
    hue="subtype",
    data=plot_data_3_sample,
    alpha=0.6,
    palette="deep",
    s=50,  # Marker size
)
plt.title("Graph 3: Slide Size vs. Number of Magnification Levels", fontsize=16)
plt.xlabel("Total Pixels at Level 0 (in hundred millions)", fontsize=12)
plt.ylabel("Number of Magnification Levels", fontsize=12)
plt.ticklabel_format(style="sci", axis="x", scilimits=(8, 8))  # Use scientific notation
plt.legend(title="Subtype", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

print("\nAnalysis complete.")
