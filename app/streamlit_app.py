import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_data, create_rfm
from src.clustering import train_clustering
from src.analytics import calculate_clv, cluster_summary
from src.recommendation import get_top_products_per_cluster

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Online Retail.csv")

CLUSTER_OFFERS = {
    0: "Premium offers — exclusive deals and early access to new products.",
    1: "Loyalty rewards — points programme and member-only discounts.",
    2: "Upsell promotions — bundle deals and product upgrade suggestions.",
    3: "Win-back discounts — limited-time discount to re-engage inactive customers.",
}

st.set_page_config(page_title="Retail Customer Segmentation", layout="wide")
st.title("Retail Customer Segmentation Dashboard")


@st.cache_data(show_spinner="Loading and processing data...")
def load_pipeline():
    df = load_data(DATA_PATH)
    rfm = create_rfm(df)
    rfm = train_clustering(rfm)
    rfm = calculate_clv(rfm)
    top_products = get_top_products_per_cluster(df, rfm)
    summary = cluster_summary(rfm)
    return df, rfm, top_products, summary


df, rfm, top_products, summary = load_pipeline()

# ── Section 1: Data Overview ──────────────────────────────────────────────────
st.subheader("Section 1 — Data Overview")

col1, col2 = st.columns(2)
col1.metric("Total Customers", f"{rfm.shape[0]:,}")
col2.metric("Total Transactions", f"{df.shape[0]:,}")

st.dataframe(df.head(10), use_container_width=True)

# ── Section 2: Cluster Distribution ──────────────────────────────────────────
st.subheader("Section 2 — Cluster Distribution")

cluster_counts = rfm["Cluster"].value_counts().sort_index()

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(
    [f"Cluster {c}" for c in cluster_counts.index],
    cluster_counts.values,
    color="#4C72B0",
)
ax2.set_xlabel("Cluster")
ax2.set_ylabel("Number of Customers")
ax2.set_title("Customers per Cluster")
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ── Section 3: Frequency vs Monetary Scatter ─────────────────────────────────
st.subheader("Section 3 — Customer Segmentation (Frequency vs Monetary)")

fig3, ax3 = plt.subplots(figsize=(8, 5))
palette = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"}
for cluster_id, group in rfm.groupby("Cluster"):
    ax3.scatter(
        group["Frequency"],
        group["Monetary"],
        label=f"Cluster {cluster_id}",
        color=palette.get(cluster_id, "#999999"),
        alpha=0.6,
        s=20,
    )
ax3.set_xlabel("Frequency")
ax3.set_ylabel("Monetary (total spend)")
ax3.set_title("Frequency vs Monetary by Cluster")
ax3.legend(title="Cluster")
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# ── Section 4: Cluster Behaviour Heatmap ─────────────────────────────────────
st.subheader("Section 4 — Cluster Behaviour Heatmap")

fig4, ax4 = plt.subplots(figsize=(8, 3))
sns.heatmap(
    summary,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    linewidths=0.5,
    ax=ax4,
)
ax4.set_title("Average RFM + CLV per Cluster")
plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

# ── Section 5: Customer Search Tool ──────────────────────────────────────────
st.subheader("Section 5 — Customer Search")

customer_id = st.number_input(
    "Enter CustomerID", min_value=0, step=1, value=0, format="%d"
)

customer_id = int(customer_id)

if customer_id and customer_id in rfm.index:
    row = rfm.loc[customer_id]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recency (days)", int(row["Recency"]))
    c2.metric("Frequency", int(row["Frequency"]))
    c3.metric("Monetary", f"${row['Monetary']:,.2f}")
    c4.metric("CLV", f"{row['CLV']:,.2f}")
    c5.metric("Cluster", int(row["Cluster"]))

    st.info(f"Marketing offer: {CLUSTER_OFFERS[int(row['Cluster'])]}")

elif customer_id != 0:
    st.warning(f"CustomerID {customer_id} not found in the dataset.")

# ── Section 6: Product Recommendations ───────────────────────────────────────
st.subheader("Section 6 — Top Product Recommendations per Cluster")

selected_cluster = st.selectbox(
    "Select a cluster to view top products",
    options=sorted(rfm["Cluster"].unique()),
    format_func=lambda x: f"Cluster {x}",
)

cluster_products = top_products[top_products["Cluster"] == selected_cluster].copy()
cluster_products = cluster_products[["Description", "TotalQuantity"]].reset_index(drop=True)
cluster_products.index += 1

st.dataframe(cluster_products, use_container_width=True)
