import pandas as pd


def calculate_clv(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add Customer Lifetime Value column.
    CLV = (Monetary * Frequency) / (Recency + 1)
    """
    rfm = rfm.copy()
    rfm["CLV"] = (rfm["Monetary"] * rfm["Frequency"]) / (rfm["Recency"] + 1)
    return rfm


def cluster_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Return mean Recency, Frequency, Monetary, and CLV per cluster.
    Requires Cluster and CLV columns to already exist in rfm.
    """
    return (
        rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary", "CLV"]]
        .mean()
        .round(2)
    )
