import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_rfm, add_additional_features
from src.segmentation_model import train_kmeans, label_segments
from src.analytics import calculate_clv, cluster_summary, cohort_analysis
from src.churn_model import create_churn_label, train_churn_model
from src.recommendation import get_top_products_per_cluster

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Online Retail.csv")

SEGMENT_ACTIONS = {
    "Champions":          "Reward them. Ask for reviews. Offer early access to new products.",
    "Loyal Customers":    "Upsell higher-value products. Enrol in loyalty programme.",
    "Potential Loyalists":"Offer membership or loyalty programme. Recommend related products.",
    "At Risk":            "Send win-back campaign. Offer personalised discount.",
}

st.set_page_config(page_title="Retail Customer Intelligence Dashboard", layout="wide")
st.title("Retail Customer Intelligence Dashboard")


# ── Pipeline (cached) ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running ML pipeline — this may take a moment...")
def run_pipeline():
    df = clean_data(load_data(DATA_PATH))
    rfm = create_rfm(df)
    rfm = add_additional_features(rfm, df)
    rfm = train_kmeans(rfm)
    rfm = label_segments(rfm)
    rfm = calculate_clv(rfm)
    rfm = create_churn_label(rfm)
    rfm = train_churn_model(rfm)
    top_products = get_top_products_per_cluster(df, rfm)
    retention = cohort_analysis(df)
    summary = cluster_summary(rfm)
    return df, rfm, top_products, retention, summary


df, rfm, top_products, retention, summary = run_pipeline()


# ── KPI Row ───────────────────────────────────────────────────────────────────
total_customers = rfm.shape[0]
total_revenue   = df["TotalPrice"].sum()
avg_clv         = rfm["CLV"].mean()
high_risk       = (rfm["ChurnProb"] > 0.7).sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers",    f"{total_customers:,}")
k2.metric("Total Revenue",      f"${total_revenue:,.0f}")
k3.metric("Average CLV",        f"{avg_clv:,.1f}")
k4.metric("High-Risk Customers (churn > 70%)", f"{high_risk:,}")

st.divider()

# ── Section 1: Segment Analysis ───────────────────────────────────────────────
st.subheader("Section 1 — Segment Analysis")

col_a, col_b = st.columns(2)

with col_a:
    seg_counts = rfm["SegmentLabel"].value_counts()
    fig1a, ax1a = plt.subplots(figsize=(6, 4))
    ax1a.bar(seg_counts.index, seg_counts.values, color="#4C72B0")
    ax1a.set_xlabel("Segment")
    ax1a.set_ylabel("Number of Customers")
    ax1a.set_title("Customer Distribution by Segment")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig1a)
    plt.close(fig1a)

with col_b:
    seg_revenue = rfm.groupby("SegmentLabel")["Monetary"].sum()
    fig1b, ax1b = plt.subplots(figsize=(6, 4))
    ax1b.pie(
        seg_revenue.values,
        labels=seg_revenue.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    )
    ax1b.set_title("Revenue Contribution by Segment")
    plt.tight_layout()
    st.pyplot(fig1b)
    plt.close(fig1b)

st.divider()

# ── Section 2: Customer Behaviour ─────────────────────────────────────────────
st.subheader("Section 2 — Customer Behaviour")

palette = {
    "Champions":          "#4C72B0",
    "Loyal Customers":    "#55A868",
    "Potential Loyalists":"#DD8452",
    "At Risk":            "#C44E52",
}

fig2, ax2 = plt.subplots(figsize=(9, 5))
for label, group in rfm.groupby("SegmentLabel"):
    ax2.scatter(
        group["Frequency"],
        group["Monetary"],
        label=label,
        color=palette.get(label, "#999999"),
        alpha=0.55,
        s=18,
    )
ax2.set_xlabel("Frequency (unique invoices)")
ax2.set_ylabel("Monetary (total spend)")
ax2.set_title("Frequency vs Monetary by Segment")
ax2.legend(title="Segment")
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.divider()

# ── Section 3: Churn Risk ─────────────────────────────────────────────────────
st.subheader("Section 3 — Churn Risk")

col_c, col_d = st.columns(2)

with col_c:
    fig3a, ax3a = plt.subplots(figsize=(6, 4))
    ax3a.hist(rfm["ChurnProb"], bins=30, color="#4C72B0", edgecolor="white")
    ax3a.set_xlabel("Churn Probability")
    ax3a.set_ylabel("Number of Customers")
    ax3a.set_title("Distribution of Churn Probability")
    plt.tight_layout()
    st.pyplot(fig3a)
    plt.close(fig3a)

with col_d:
    fig3b, ax3b = plt.subplots(figsize=(6, 4))
    sc = ax3b.scatter(
        rfm["Recency"],
        rfm["CLV"],
        c=rfm["ChurnProb"],
        cmap="RdYlGn_r",
        alpha=0.6,
        s=18,
    )
    plt.colorbar(sc, ax=ax3b, label="Churn Probability")
    ax3b.set_xlabel("Recency (days)")
    ax3b.set_ylabel("CLV")
    ax3b.set_title("CLV vs Recency (coloured by Churn Risk)")
    plt.tight_layout()
    st.pyplot(fig3b)
    plt.close(fig3b)

st.divider()

# ── Section 4: Cohort Retention ───────────────────────────────────────────────
st.subheader("Section 4 — Cohort Retention")

# Limit to first 12 months for readability
retention_display = retention.iloc[:, :12]

fig4, ax4 = plt.subplots(figsize=(14, max(6, len(retention_display) * 0.55)))
sns.heatmap(
    retention_display.astype(float),
    annot=True,
    fmt=".0%",
    cmap="Blues",
    linewidths=0.3,
    ax=ax4,
    cbar_kws={"label": "Retention Rate"},
)
ax4.set_title("Monthly Cohort Retention")
ax4.set_xlabel("Months Since First Purchase")
ax4.set_ylabel("Cohort Month")
plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

st.divider()

# ── Section 5: Product Insights ───────────────────────────────────────────────
st.subheader("Section 5 — Top Products per Segment")

segments = sorted(rfm["SegmentLabel"].unique())
selected_seg = st.selectbox("Select a segment", options=segments)

seg_products = top_products[top_products["SegmentLabel"] == selected_seg].copy()
seg_products = seg_products[["Description", "TotalQuantity"]].reset_index(drop=True)
seg_products.index += 1
st.dataframe(seg_products, use_container_width=True)

st.divider()

# ── Section 6: Customer Search ────────────────────────────────────────────────
st.subheader("Section 6 — Customer Search")

customer_id = int(st.number_input("Enter CustomerID", min_value=0, step=1, value=0, format="%d"))

if customer_id and customer_id in rfm.index:
    row = rfm.loc[customer_id]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Recency (days)",    int(row["Recency"]))
    c2.metric("Frequency",         int(row["Frequency"]))
    c3.metric("Monetary",          f"${row['Monetary']:,.2f}")
    c4.metric("CLV",               f"{row['CLV']:,.1f}")
    c5.metric("Segment",           str(row["SegmentLabel"]))
    c6.metric("Churn Probability", f"{row['ChurnProb']:.1%}")

    action = SEGMENT_ACTIONS.get(str(row["SegmentLabel"]), "No action defined.")
    st.info(f"Recommended action: {action}")

elif customer_id != 0:
    st.warning(f"CustomerID {customer_id} not found in the dataset.")
