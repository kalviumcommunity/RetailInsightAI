import pandas as pd
from sklearn.cluster import KMeans


def load_data(path):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(path)
    return df


def prepare_features(df):
    """
    Select features used for clustering
    """
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    return X


def run_kmeans(X, n_clusters=5):
    """
    Train KMeans clustering model
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X)
    return model, clusters


def add_clusters(df, clusters):
    """
    Attach cluster labels to dataset
    """
    df["Cluster"] = clusters
    return df