import pandas as pd


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the Online Retail dataset and perform basic cleaning."""
    df = pd.read_csv(path, encoding="ISO-8859-1")

    # Drop rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove negative or zero quantities (returns/cancellations)
    df = df[df["Quantity"] > 0]

    # Remove negative or zero unit prices
    df = df[df["UnitPrice"] > 0]

    # Create TotalPrice column
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Ensure CustomerID is integer
    df["CustomerID"] = df["CustomerID"].astype(int)

    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate RFM (Recency, Frequency, Monetary) features.
    Recency  - days since last purchase (lower = better)
    Frequency - number of unique invoices
    Monetary  - total spend
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
