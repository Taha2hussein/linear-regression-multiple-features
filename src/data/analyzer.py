from pandas import DataFrame

BYTES_PER_MB = 1024 ** 2


def preview(df: DataFrame, rows: int = 5) -> DataFrame:
    """
    Return the first N rows from the dataset.
    """
    return df.head(rows)

def dataset_summary(df: DataFrame) -> dict:
    """
    Return general information about the dataset.
    """
    return {
        "rows_count": df.shape[0],
        "columns_count": df.shape[1],
        "memory_usage_mb": round(
           df.memory_usage(deep=True).sum() / BYTES_PER_MB,
           2
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum())
    }
    
def missing_values_report(df: DataFrame) -> DataFrame:
    """
    Return missing values count and percentage for every column.
    """
    report = DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing %": round(
            (df.isna().sum() / len(df)) * 100,
            2
        )
    })
    return report[
       report["Missing Count"] > 0
    ]
    
def duplicate_report(df: DataFrame) -> dict:
    """
    Return duplicate rows statistics.
    """
    duplicates = int(df.duplicated().sum())
    percentage = (
    round((duplicates / len(df)) * 100, 2)
    if len(df) > 0
    else 0
    )
    return {
       "duplicate_rows": duplicates,
       "duplicate_percentage": percentage
}

def numeric_columns(df: DataFrame) -> list[str]:
    """
    Return all numeric columns.
    """
    return df.select_dtypes(
        include="number"
    ).columns.tolist()
    
def categorical_columns(df: DataFrame) -> list[str]:
    """
    Return all categorical columns.
    """
    return df.select_dtypes(
        exclude="number"
    ).columns.tolist()
    