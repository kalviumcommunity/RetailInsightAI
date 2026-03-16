import pandas as pd


def get_top_products_per_cluster(
    df: pd.DataFrame, rfm: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    """
    Join cluster labels back to the transaction data and return
    the top N most purchased products per cluster.

    Parameters
    ----------
    df  : cleaned transaction dataframe (from preprocessing)
    rfm : RFM dataframe with a Cluster column
    top_n : number of top products to return per cluster

    Returns
    -------
    DataFrame with columns: Cluster, Description, TotalQuantity
    """
    # Merge cluster labels into transactions
    df_merged = df.merge(
        rfm[["Cluster"]],
        left_on="CustomerID",
        right_index=True,
        how="inner",
    )

    # Aggregate quantity sold per cluster + product
    product_agg = (
        df_merged.groupby(["Cluster", "Description"])["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "TotalQuantity"})
    )

    # Pick top N per cluster
    top_products = (
        product_agg.sort_values(["Cluster", "TotalQuantity"], ascending=[True, False])
        .groupby("Cluster")
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_products
