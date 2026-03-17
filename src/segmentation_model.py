import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

# Human-readable labels assigned after inspecting cluster centroids.
# Champions  = high frequency, high monetary, low recency
# Loyal       = moderate-high frequency, moderate monetary
# Potential   = moderate frequency, lower monetary
# At Risk     = high recency (haven't bought recently)
SEGMENT_LABELS = {
    0: "Champions",
    1: "At Risk",
    2: "Loyal Customers",
    3: "Potential Loyalists",
}


def train_kmeans(rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Scale RFM features, train KMeans, attach cluster labels,
    and persist model artefacts to model/.

    Returns the RFM DataFrame with a Cluster column.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    return rfm


def label_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable segment labels based on cluster centroids.
    Clusters are ranked by a composite score:
      score = -Recency + Frequency + Monetary (all normalised 0-1)
    so the highest-scoring cluster becomes 'Champions', etc.
    """
    rfm = rfm.copy()

    # Normalise each RFM dimension to [0, 1] for ranking
    for col in ["Recency", "Frequency", "Monetary"]:
        col_min = rfm[col].min()
        col_max = rfm[col].max()
        if col_max > col_min:
            rfm[f"_{col}_norm"] = (rfm[col] - col_min) / (col_max - col_min)
        else:
            rfm[f"_{col}_norm"] = 0.0

    # Composite score: low recency is good, high freq/monetary is good
    rfm["_score"] = (
        (1 - rfm["_Recency_norm"]) + rfm["_Frequency_norm"] + rfm["_Monetary_norm"]
    )

    # Rank clusters by mean score descending
    cluster_scores = rfm.groupby("Cluster")["_score"].mean().sort_values(ascending=False)
    rank_to_label = {
        0: "Champions",
        1: "Loyal Customers",
        2: "Potential Loyalists",
        3: "At Risk",
    }
    cluster_to_label = {
        cluster: rank_to_label[rank]
        for rank, cluster in enumerate(cluster_scores.index)
    }

    rfm["SegmentLabel"] = rfm["Cluster"].map(cluster_to_label)

    # Drop temp columns
    rfm.drop(columns=["_Recency_norm", "_Frequency_norm", "_Monetary_norm", "_score"],
             inplace=True)

    return rfm
