import numpy as np
import pandas as pd


def _map_loyalty_category(tenure: float) -> str:
    if tenure <= 12:
        return "New"
    if tenure <= 24:
        return "Regular"
    return "Loyal"


def _map_recent_activity(days_since_last_order: float) -> str:
    if days_since_last_order <= 7:
        return "Very Active"
    if days_since_last_order <= 30:
        return "Active"
    if days_since_last_order <= 60:
        return "At Risk"
    return "Dormant"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-driven features for churn prediction."""
    df = df.copy()
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
    df["ordercount"] = pd.to_numeric(df["ordercount"], errors="coerce").fillna(0)
    df["cashbackamount"] = pd.to_numeric(df["cashbackamount"], errors="coerce").fillna(0)
    df["warehousetohome"] = pd.to_numeric(df["warehousetohome"], errors="coerce").fillna(0)
    df["daysincelastorder"] = pd.to_numeric(df["daysincelastorder"], errors="coerce").fillna(0)
    df["hourspendonapp"] = pd.to_numeric(df["hourspendonapp"], errors="coerce").fillna(0)

    df["orderfrequency"] = np.where(
        df["tenure"] > 0,
        df["ordercount"] / df["tenure"],
        df["ordercount"],
    )
    df["customervaluescore"] = df["ordercount"] * df["cashbackamount"]
    df["loyaltycategory"] = df["tenure"].apply(_map_loyalty_category)
    distance_median = df["warehousetohome"].median()
    df["highdistancecustomer"] = (df["warehousetohome"] > distance_median).astype(int)
    df["complaintflag"] = df["complaint"].astype(str).str.lower().isin(
        ["yes", "y", "1", "true", "complaint"]
    ).astype(int)
    df["recentcustomeractivity"] = df["daysincelastorder"].apply(_map_recent_activity)
    df["engagementscore"] = df["hourspendonapp"] * df["ordercount"]

    return df
