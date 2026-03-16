import pandas as pd


def calculate_clv(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Customer Lifetime Value column.
    CLV = (Monetary * Frequency) / (Recency + 1)
    """
    rfm = rfm.copy()
    rfm["CLV"] = (rfm["Monetary"] * rfm["Frequency"]) / (rfm["Recency"] + 1)
    return rfm


def cluster_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Return average Recency, Frequency, Monetary, and CLV per cluster.
    Requires the RFM dataframe to already have Cluster and CLV columns.
    """
    return (
        rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary", "CLV"]]
        .mean()
        .round(2)
    )
