import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import clean_data
from src.feature_engineering import create_rfm, add_additional_features
from src.segmentation_model import train_kmeans, label_segments
from src.analytics import calculate_clv, cohort_analysis
from src.churn_model import create_churn_label, train_churn_model
from src.recommendation import get_top_products_per_cluster

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Customer Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Force white background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    [data-testid="stHeader"] { background: #ffffff !important; box-shadow: none; }

    /* Main content padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

    /* Section headings */
    .section-heading {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #4a90d9;
        padding-bottom: 6px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* KPI cards */
    .kpi-box {
        background: #f7f8fa;
        border: 1px solid #e4e4e4;
        border-radius: 10px;
        padding: 20px 16px;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    /* Hide default streamlit top padding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Pipeline ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full ML pipeline and return (rfm, df_clean, cohort_retention)."""
    raw = pd.read_csv(io.BytesIO(file_bytes), encoding="ISO-8859-1")
    df = clean_data(raw)

    rfm = create_rfm(df)
    rfm = add_additional_features(rfm, df)
    rfm = train_kmeans(rfm, n_clusters=4)
    rfm = label_segments(rfm)
    rfm = calculate_clv(rfm)
    rfm = create_churn_label(rfm)
    rfm = train_churn_model(rfm)

    retention = cohort_analysis(df)
    return rfm, df, retention

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0;">
    <div style="font-size: 0.8rem; color: #4a90d9; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px;">
        Analytics Platform
    </div>
    <div style="font-size: 2rem; font-weight: 800; color: #1a1a2e; line-height: 1.2;">
        Retail Customer Intelligence Dashboard
    </div>
    <div style="font-size: 0.9rem; color: #666; margin-top: 8px;">
        Upload a retail transactions CSV to run the full analytics pipeline.
    </div>
</div>
<hr style="border: none; border-top: 1px solid #e8e8e8; margin: 1.2rem 0 1.5rem 0;">
""", unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to get started.")
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
with st.spinner("Running pipeline — this may take a moment on first run..."):
    try:
        rfm, df_clean, retention = run_pipeline(uploaded_file.read())
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

required_cols = {"Recency", "Frequency", "Monetary", "CLV", "ChurnProb", "SegmentLabel"}
missing = required_cols - set(rfm.columns)
if missing:
    st.error(f"Missing expected columns after pipeline: {missing}")
    st.stop()

# ── KPI Section ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Key Metrics</div>', unsafe_allow_html=True)

total_customers = len(rfm)
total_revenue = df_clean["TotalPrice"].sum()
avg_clv = rfm["CLV"].mean()
high_risk = (rfm["ChurnProb"] > 0.7).sum()

col1, col2, col3, col4 = st.columns(4)
for col, label, value in [
    (col1, "Total Customers", f"{total_customers:,}"),
    (col2, "Total Revenue", f"£{total_revenue:,.0f}"),
    (col3, "Average CLV", f"£{avg_clv:,.2f}"),
    (col4, "High Risk Customers", f"{high_risk:,}"),
]:
    col.markdown(
        f'<div class="kpi-box"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

# ── Section 1 — Segment Analysis ─────────────────────────────────────────────
st.markdown('<div class="section-heading">Section 1 — Segment Analysis</div>', unsafe_allow_html=True)

seg_counts = rfm["SegmentLabel"].value_counts().reset_index()
seg_counts.columns = ["Segment", "Customers"]

seg_revenue = (
    rfm.groupby("SegmentLabel")["Monetary"].sum().reset_index()
    .rename(columns={"SegmentLabel": "Segment", "Monetary": "Revenue"})
)

col_a, col_b = st.columns(2)

with col_a:
    fig_bar = px.bar(
        seg_counts, x="Segment", y="Customers",
        color="Segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Customers per Segment",
    )
    fig_bar.update_layout(showlegend=False, plot_bgcolor="#fff", paper_bgcolor="#fff",
                          font_color="#333", title_font_size=14)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    fig_pie = px.pie(
        seg_revenue, names="Segment", values="Revenue",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Revenue Contribution per Segment",
    )
    fig_pie.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff",
                          font_color="#333", title_font_size=14)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Section 2 — Customer Behavior ────────────────────────────────────────────
st.markdown('<div class="section-heading">Section 2 — Customer Behavior</div>', unsafe_allow_html=True)

fig_scatter = px.scatter(
    rfm.reset_index(), x="Frequency", y="Monetary",
    color="SegmentLabel",
    color_discrete_sequence=px.colors.qualitative.Set2,
    opacity=0.6,
    title="Frequency vs Monetary by Segment",
    labels={"Frequency": "Purchase Frequency", "Monetary": "Total Spend (£)"},
    hover_data=["CustomerID"] if "CustomerID" in rfm.reset_index().columns else None,
)
fig_scatter.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff",
                           font_color="#333", title_font_size=14)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Section 3 — Churn Analysis ────────────────────────────────────────────────
st.markdown('<div class="section-heading">Section 3 — Churn Analysis</div>', unsafe_allow_html=True)

col_c, col_d = st.columns(2)

with col_c:
    fig_hist = px.histogram(
        rfm, x="ChurnProb", nbins=30,
        title="Churn Probability Distribution",
        labels={"ChurnProb": "Churn Probability"},
        color_discrete_sequence=["#4a90d9"],
    )
    fig_hist.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff",
                            font_color="#333", title_font_size=14)
    st.plotly_chart(fig_hist, use_container_width=True)

with col_d:
    rfm_plot = rfm.copy()
    rfm_plot["Risk"] = rfm_plot["ChurnProb"].apply(lambda x: "High Risk" if x > 0.7 else "Low Risk")
    fig_clv = px.scatter(
        rfm_plot.reset_index(), x="CLV", y="Recency",
        color="Risk",
        color_discrete_map={"High Risk": "#e05c5c", "Low Risk": "#4a90d9"},
        opacity=0.6,
        title="CLV vs Recency (Risk Level)",
        labels={"CLV": "Customer Lifetime Value", "Recency": "Recency (days)"},
    )
    fig_clv.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff",
                           font_color="#333", title_font_size=14)
    st.plotly_chart(fig_clv, use_container_width=True)

# ── Section 4 — Retention Analysis ───────────────────────────────────────────
st.markdown('<div class="section-heading">Section 4 — Retention Analysis</div>', unsafe_allow_html=True)

try:
    retention_display = retention.copy()
    retention_display.index = retention_display.index.astype(str)
    retention_display.columns = [str(c) for c in retention_display.columns]

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=retention_display.values,
        x=[f"Month {c}" for c in retention_display.columns],
        y=retention_display.index.tolist(),
        colorscale=[[0, "#f0f4ff"], [1, "#1a4fa0"]],
        text=np.round(retention_display.values * 100, 1),
        texttemplate="%{text}%",
        showscale=True,
        hoverongaps=False,
    ))
    fig_heatmap.update_layout(
        title="Monthly Cohort Retention Rate",
        xaxis_title="Months Since First Purchase",
        yaxis_title="Cohort Month",
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font_color="#333",
        title_font_size=14,
        height=500,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render cohort heatmap: {e}")

# ── Section 5 — Product Insights ─────────────────────────────────────────────
st.markdown('<div class="section-heading">Section 5 — Product Insights</div>', unsafe_allow_html=True)

try:
    top_products = get_top_products_per_cluster(df_clean, rfm, top_n=5)
    segments = top_products["SegmentLabel"].unique()
    cols = st.columns(min(len(segments), 4))
    for i, seg in enumerate(sorted(segments)):
        with cols[i % len(cols)]:
            st.markdown(f"**{seg}**")
            seg_df = (
                top_products[top_products["SegmentLabel"] == seg][["Description", "TotalQuantity"]]
                .reset_index(drop=True)
            )
            seg_df.index += 1
            st.dataframe(seg_df, use_container_width=True, hide_index=False)
except Exception as e:
    st.warning(f"Could not load product insights: {e}")

# ── Section 6 — Customer Search ───────────────────────────────────────────────
st.markdown('<div class="section-heading">Section 6 — Customer Search</div>', unsafe_allow_html=True)

rfm_indexed = rfm.reset_index()
customer_ids = rfm_indexed["CustomerID"].tolist()

customer_id = st.number_input(
    "Enter Customer ID", min_value=0, step=1, value=0,
    help="Enter a valid CustomerID from the dataset"
)

RECOMMENDATIONS = {
    "Champions": "Reward them — offer loyalty perks or early access to new products.",
    "Loyal Customers": "Upsell — suggest premium products or bundle deals.",
    "Potential Loyalists": "Nurture — send personalised offers to increase engagement.",
    "At Risk": "Re-engage — send a win-back campaign with a discount or reminder.",
}

if customer_id != 0:
    match = rfm_indexed[rfm_indexed["CustomerID"] == customer_id]
    if match.empty:
        st.error(f"Customer ID {customer_id} not found in the dataset.")
    else:
        row = match.iloc[0]
        st.markdown(f'<div style="font-size:1rem;font-weight:700;color:#1a1a2e;margin:1rem 0 0.5rem 0;">Profile for Customer {int(customer_id)}</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m4, m5, m6 = st.columns(3)

        m1.metric("Recency (days)", f"{int(row['Recency'])}")
        m2.metric("Frequency", f"{int(row['Frequency'])}")
        m3.metric("Monetary", f"£{row['Monetary']:,.2f}")
        m4.metric("CLV", f"£{row['CLV']:,.2f}")
        m5.metric("Segment", row["SegmentLabel"])
        m6.metric("Churn Probability", f"{row['ChurnProb']:.1%}")

        segment = row["SegmentLabel"]
        rec_text = RECOMMENDATIONS.get(segment, "No recommendation available.")
        st.markdown(
            f'<div style="background:#f0f4ff;border-left:4px solid #4a90d9;'
            f'padding:12px 16px;border-radius:4px;margin-top:12px;">'
            f'<strong>Recommendation ({segment}):</strong> {rec_text}</div>',
            unsafe_allow_html=True,
        )
