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
    Return mean Recency, Frequency, Monetary, CLV, and ChurnProb per segment.
    """
    cols = [c for c in ["Recency", "Frequency", "Monetary", "CLV", "ChurnProb"]
            if c in rfm.columns]
    return rfm.groupby("SegmentLabel")[cols].mean().round(2)


def cohort_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a monthly cohort retention table.

    Returns a pivot table where:
      rows    = cohort month (first purchase month)
      columns = months since first purchase (0, 1, 2, ...)
      values  = retention rate (fraction of cohort still active)
    """
    df = df.copy()

    # Cohort month = month of customer's first purchase
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")
    cohort_month = (
        df.groupby("CustomerID")["InvoiceMonth"]
        .min()
        .rename("CohortMonth")
    )
    df = df.join(cohort_month, on="CustomerID")

    # Months since cohort start
    df["MonthsElapsed"] = (
        df["InvoiceMonth"] - df["CohortMonth"]
    ).apply(lambda x: x.n)

    # Count unique customers per cohort + elapsed month
    cohort_data = (
        df.groupby(["CohortMonth", "MonthsElapsed"])["CustomerID"]
        .nunique()
        .reset_index()
    )

    cohort_pivot = cohort_data.pivot_table(
        index="CohortMonth", columns="MonthsElapsed", values="CustomerID"
    )

    # Retention rate relative to cohort size (month 0)
    cohort_sizes = cohort_pivot[0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0).round(3)

    return retention
