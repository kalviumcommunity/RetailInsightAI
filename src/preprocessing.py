import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load the Online Retail dataset.
    Supports both .csv and .xlsx formats.
    Cleans the data:
      - removes rows with missing CustomerID
      - removes rows with Quantity <= 0
      - removes rows with UnitPrice <= 0
      - creates TotalPrice column
      - converts InvoiceDate to datetime
    """
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="ISO-8859-1")

    # Drop rows with missing CustomerID
    df.dropna(subset=["CustomerID"], inplace=True)

    # Remove negative/zero quantities (returns, cancellations)
    df = df[df["Quantity"] > 0]

    # Remove negative/zero unit prices
    df = df[df["UnitPrice"] > 0]

    # Create TotalPrice column
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Ensure CustomerID is integer
    df["CustomerID"] = df["CustomerID"].astype(int)

    return df


def create_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM features per customer.
      Recency  - days since last purchase (lower = more recent)
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
