from pandas import DataFrame


def transform_column_types(df: DataFrame) -> DataFrame:
    """
    Transform specific columns in the DataFrame to appropriate data types.
    - "Engine": Remove "CC" and convert to float.
    - "Max Power": Remove "bhp" and convert to float.
    - "Max Torque": Remove "Nm" and convert to float.
    """
    df = df.copy()

    # Engine
    df["Engine"] = (
    df["Engine"]
    .str.replace("cc", "", case=False, regex=True)
    .str.strip()
    .astype(float)
)

    # Max Power
    df["Max Power"] = (
        df["Max Power"]
        .str.extract(r"(\d+\.?\d*)\s*bhp")
        .astype(float)
    )

    # Max Torque
    df["Max Torque"] = (
        df["Max Torque"]
        .str.extract(r"(\d+\.?\d*)\s*Nm")
        .astype(float)
    )

    return df