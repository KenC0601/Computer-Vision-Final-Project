from sklearn.datasets import fetch_openml
import pandas as pd

print("Attempting to fetch dataset 44282 using sklearn...")
try:
    # Note: fetch_openml usually downloads ARFF. 
    # If the dataset is only available as Parquet on OpenML, this might fail or return metadata only.
    dataset = fetch_openml(data_id=44282, as_frame=True, parser='auto')
    print("Fetch successful.")
    print(f"Data shape: {dataset.data.shape}")
    print(f"Columns: {dataset.data.columns}")
except Exception as e:
    print(f"Fetch failed: {e}")
