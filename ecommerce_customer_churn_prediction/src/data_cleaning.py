import pandas as pd

NUMERIC_COLUMNS = [
    "tenure",
    "warehousetohome",
    "hourspendonapp",
    "numberofdeviceregistered",
    "satisfactionscore",
    "numberofaddress",
    "orderamounthikefromlastyear",
    "couponused",
    "ordercount",
    "daysincelastorder",
    "cashbackamount",
    "churn",
]

CATEGORICAL_COLUMNS = [
    "preferredlogindevice",
    "citytier",
    "preferredpaymentmode",
    "gender",
    "preferedordercat",
    "maritalstatus",
    "complaint",
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load the churn dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw churn data, impute missing values, and normalize columns."""
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]
    df = df.rename(columns={"complain": "complaint"})

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            replacement = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].replace({"nan": None, "none": None})
            df[col] = df[col].fillna(replacement)

    df = df.drop_duplicates().reset_index(drop=True)

    if "customerid" in df.columns:
        df["customerid"] = df["customerid"].astype(str)
    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(int)

    return df


def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for the dataset."""
    return df.describe(include="all").transpose()
