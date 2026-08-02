from pandas import DataFrame


# ------------------------
# Find Outliers (IQR)
# ------------------------
def detect_outliers_iqr(df: DataFrame, feature: str) -> dict:
    series = df[feature]

    q1, q3 = series.dropna().quantile([0.25, 0.75])
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (
        series.notna() &
        ~series.between(lower_bound, upper_bound)
    )

    outliers = df[mask]

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers_count": len(outliers),
        "outliers": outliers,
    }
    
def outliers_report(df: DataFrame, feature: str) -> dict:
    result = detect_outliers_iqr(df, feature)

    outliers = result["outliers"][feature]
    count = len(outliers)

    return {
        "feature": feature,
        "outliers_count": count,
        "outliers_percentage": round(count / len(df) * 100, 2),
        "lower_bound": result["lower_bound"],
        "upper_bound": result["upper_bound"],
        "min_outlier": outliers.min(),
        "max_outlier": outliers.max(),
        "mean_outlier": round(outliers.mean(), 2),
        "median_outlier": outliers.median(),
    }