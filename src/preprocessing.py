import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Read the Online Retail dataset from disk.
    Supports .csv (ISO-8859-1) and .xlsx formats.
    Returns a raw DataFrame — call clean_data() next.
    """
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="ISO-8859-1")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Online Retail DataFrame:
      - Remove rows with missing CustomerID
      - Remove rows with Quantity <= 0  (returns / cancellations)
      - Remove rows with UnitPrice <= 0
      - Create TotalPrice = Quantity * UnitPrice
      - Convert InvoiceDate to datetime
      - Cast CustomerID to int
    """
    df = df.copy()
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df
