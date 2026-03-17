import pandas as pd


def get_top_products_per_cluster(
    df: pd.DataFrame, rfm: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    """
    Merge cluster/segment labels into the transaction dataset and return
    the top N most purchased products per segment.

    Parameters
    ----------
    df    : cleaned transaction DataFrame (from load_data)
    rfm   : RFM DataFrame with SegmentLabel column
    top_n : number of top products per segment (default 5)

    Returns
    -------
    DataFrame with columns: SegmentLabel, Description, TotalQuantity
    """
    label_col = "SegmentLabel" if "SegmentLabel" in rfm.columns else "Cluster"

    df_merged = df.merge(
        rfm[[label_col]],
        left_on="CustomerID",
        right_index=True,
        how="inner",
    )

    product_agg = (
        df_merged.groupby([label_col, "Description"])["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "TotalQuantity"})
    )

    top_products = (
        product_agg
        .sort_values([label_col, "TotalQuantity"], ascending=[True, False])
        .groupby(label_col)
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_products
