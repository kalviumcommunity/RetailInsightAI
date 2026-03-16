import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.clustering import load_data, prepare_features, run_kmeans, add_clusters


st.title("Retail Customer Segmentation Dashboard")

uploaded_file = st.file_uploader("Upload Customer Dataset", type=["csv"])


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    X = prepare_features(df)

    model, clusters = run_kmeans(X)

    df = add_clusters(df, clusters)

    st.write("Clustered Dataset")
    st.dataframe(df.head())

    fig, ax = plt.subplots()

    ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )

    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Segments")

    st.pyplot(fig)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Clustered Data", csv, "clustered_data.csv", "text/csv")