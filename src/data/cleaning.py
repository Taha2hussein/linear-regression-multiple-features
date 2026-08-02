from pandas import DataFrame
from pandas.api.types import is_numeric_dtype


def remove_duplicates(df: DataFrame) -> DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates()

def fill_missing_values(df: DataFrame) -> DataFrame:
    """
    Fill missing values in the DataFrame with appropriate strategies.
    - Numeric columns: Fill with the median.
    - Categorical columns: Fill with the mode.
    """
    for column in df.columns:
        if  is_numeric_dtype(df[column]):
             df[column] = df[column].fillna(df[column].median())
        else:
            mode = df[column].mode()

            if not mode.empty:
                df[column] = df[column].fillna(mode[0])

    return df
