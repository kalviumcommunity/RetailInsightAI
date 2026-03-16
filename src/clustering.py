import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def train_clustering(rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Scale RFM features with StandardScaler, train KMeans,
    attach cluster labels, and save model artefacts.

    Saves:
      model/kmeans_model.pkl
      model/scaler.pkl

    Returns the RFM DataFrame with a Cluster column.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    return rfm


def load_and_predict(rfm: pd.DataFrame) -> pd.DataFrame:
    """Load saved models and assign cluster labels to an RFM DataFrame."""
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))

    rfm_scaled = scaler.transform(rfm[["Recency", "Frequency", "Monetary"]])
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.predict(rfm_scaled)
    return rfm
