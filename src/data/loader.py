from pandas import read_csv, DataFrame
from pathlib import Path

def load_data(path: Path)-> DataFrame :
    """
    Load data from a CSV file.
    """
    
    if  not path.exists():
        raise FileNotFoundError(f"Path does not exist at path {path}")
    return read_csv(path)