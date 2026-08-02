from pandas import DataFrame


def get_target_correlation(df: DataFrame, target: str):
    """
    Calculate correlation matrix for numeric columns in the DataFrame.
    """
    return df.corr(numeric_only=True)[target].drop(target).sort_values(
        key=abs,
        ascending=False
)
    
def get_top_correlated_features(df: DataFrame, target: str, count: int = 5):
    """
    Return the top N features with the highest absolute
    correlation to the target column.
    """
    
    correlated_series = get_target_correlation(df, target)
    return correlated_series.head(count)