import pandas as pd
import os
import kagglehub

def run_task_1_1():
    # Download dataset (if not already downloaded)
    path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
    csv_path = os.path.join(path, "WineQT.csv")

    # Load into DataFrame
    df = pd.read_csv(csv_path)

    # Q1.1.1 - First 5 rows, info, stats
    print("First 5 rows:")
    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    # Q1.1.2 - Feature with highest variation
    variation = df.std().sort_values(ascending=False)
    print("\nFeatures sorted by variation:")
    print(variation)

    return df
