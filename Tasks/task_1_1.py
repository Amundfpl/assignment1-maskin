import pandas as pd
import os
import kagglehub

def run_task_1_1(df):
    # Download dataset (if not already downloaded)

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
