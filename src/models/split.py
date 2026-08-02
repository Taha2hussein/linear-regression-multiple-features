from sklearn.model_selection import train_test_split
from pandas import DataFrame

def split_data(df: DataFrame, target: str, test_size: float = 0.20, random_state: int = 42):
    X = df.drop(columns = [target])
    y = df[target]
    return train_test_split(X,y, test_size=test_size, random_state=random_state)
    