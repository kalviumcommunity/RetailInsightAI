import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
CHURN_THRESHOLD = 90  # days — customers inactive longer than this are labelled churned


def create_churn_label(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary Churned column.
    Churned = 1 if Recency > CHURN_THRESHOLD, else 0.
    """
    rfm = rfm.copy()
    rfm["Churned"] = (rfm["Recency"] > CHURN_THRESHOLD).astype(int)
    return rfm


def train_churn_model(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Train a RandomForestClassifier to predict churn probability.
    Features: Recency, Frequency, Monetary, AvgOrderValue, PurchaseInterval
    Target:   Churned

    Saves model to model/churn_model.pkl.
    Returns the RFM DataFrame with a ChurnProb column added.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    feature_cols = [c for c in
                    ["Recency", "Frequency", "Monetary", "AvgOrderValue", "PurchaseInterval"]
                    if c in rfm.columns]

    X = rfm[feature_cols]
    y = rfm["Churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    joblib.dump(clf, os.path.join(MODEL_DIR, "churn_model.pkl"))

    rfm = rfm.copy()
    rfm["ChurnProb"] = clf.predict_proba(X)[:, 1]

    return rfm
