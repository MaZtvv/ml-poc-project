from pathlib import Path

import pandas as pd


EXTERNAL_DATA_DIR = "/Users/maxime/Documents/Projet/POC ML/data"


def list_raw_files(raw_dir="data/raw"):
    path = Path(raw_dir)
    if not path.exists():
        return []
    return sorted(file for file in path.iterdir() if file.is_file())


def list_external_data_files(external_dir=EXTERNAL_DATA_DIR):
    path = Path(external_dir)
    if not path.exists():
        return []
    return sorted(file for file in path.iterdir() if file.is_file())


def load_csv_dataset(path):
    return pd.read_csv(path)
