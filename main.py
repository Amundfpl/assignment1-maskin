import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
print("Path to dataset files:", path)

# Path to CSV file
csv_path = os.path.join(path, "WineQT.csv")  # adjust filename if needed

# Load into DataFrame
df = pd.read_csv(csv_path)

# 1. Display first 5 rows
print("First 5 rows:")
print(df.head())

# 2. Print dataset info
print("\nDataset Information:")
print(df.info())

# 3. Print summary statistics
print("\nSummary Statistics:")
print(df.describe())

print("Features with highest standard deviation:")
print(df.describe().T[["std"]].sort_values("std", ascending=False))
