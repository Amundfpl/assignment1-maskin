import pandas as pd
import os
import kagglehub

def load_dataset():
    """
    Downloads the Wine Quality dataset (if not already downloaded),
    loads it into a pandas DataFrame, and returns it.
    """
    path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
    csv_path = os.path.join(path, "WineQT.csv")
    df = pd.read_csv(csv_path)
    return df
