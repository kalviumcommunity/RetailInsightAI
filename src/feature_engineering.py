import pandas as pd
import numpy as np


def create_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM features per customer.
      Recency   - days since last purchase (lower = more recent)
      Frequency - number of unique invoices
      Monetary  - total spend
    Returns a DataFrame indexed by CustomerID.
    """
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
        .set_index("CustomerID")
    )

    return rfm


def add_additional_features(rfm: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the RFM DataFrame with:
      AvgOrderValue     - Monetary / Frequency
      PurchaseInterval  - average days between purchases (0 if only 1 purchase)
    """
    rfm = rfm.copy()

    # Average Order Value
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]

    # Average days between purchases per customer
    def avg_interval(dates):
        sorted_dates = sorted(dates)
        if len(sorted_dates) < 2:
            return 0.0
        gaps = [(sorted_dates[i + 1] - sorted_dates[i]).days
                for i in range(len(sorted_dates) - 1)]
        return float(np.mean(gaps))

    intervals = (
        df.groupby("CustomerID")["InvoiceDate"]
        .apply(avg_interval)
        .rename("PurchaseInterval")
    )

    rfm = rfm.join(intervals)
    rfm["PurchaseInterval"] = rfm["PurchaseInterval"].fillna(0.0)

    return rfm
