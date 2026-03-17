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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Customer Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BLUE        = "#2563EB"
BLUE_LIGHT  = "#EFF6FF"
BLUE_MID    = "#BFDBFE"
RED         = "#DC2626"
RED_LIGHT   = "#FEF2F2"
GRAY_900    = "#111827"
GRAY_700    = "#374151"
GRAY_500    = "#6B7280"
GRAY_200    = "#E5E7EB"
GRAY_100    = "#F9FAFB"
WHITE       = "#FFFFFF"

# Segment palette — 4 distinct, accessible blues/neutrals
SEG_COLORS = {
    "Champions":          "#1D4ED8",
    "Loyal Customers":    "#2563EB",
    "Potential Loyalists":"#60A5FA",
    "At Risk":            "#DC2626",
}
SEG_COLOR_LIST = list(SEG_COLORS.values())

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Reset & base ── */
  .stApp, [data-testid="stAppViewContainer"],
  [data-testid="stHeader"], section[data-testid="stSidebar"] {{
      background-color: {WHITE} !important;
  }}
  [data-testid="stHeader"] {{ box-shadow: none !important; }}
  .block-container {{
      padding: 2rem 3rem 3rem 3rem !important;
      max-width: 1280px !important;
  }}
  #MainMenu, footer {{ visibility: hidden; }}

  /* ── Page header ── */
  .page-header {{
      padding: 0.5rem 0 1.5rem 0;
      border-bottom: 2px solid {GRAY_200};
      margin-bottom: 2rem;
  }}
  .page-eyebrow {{
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: {BLUE};
      margin-bottom: 6px;
  }}
  .page-title {{
      font-size: 1.85rem;
      font-weight: 800;
      color: {GRAY_900};
      line-height: 1.2;
      margin: 0;
  }}
  .page-subtitle {{
      font-size: 0.88rem;
      color: {GRAY_500};
      margin-top: 6px;
  }}

  /* ── Section header ── */
  .section-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 2.5rem 0 1.2rem 0;
  }}
  .section-number {{
      background: {BLUE};
      color: {WHITE};
      font-size: 0.7rem;
      font-weight: 700;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
  }}
  .section-title {{
      font-size: 1rem;
      font-weight: 700;
      color: {GRAY_900};
      margin: 0;
  }}
  .section-desc {{
      font-size: 0.8rem;
      color: {GRAY_500};
      margin: 0;
  }}

  /* ── KPI cards ── */
  .kpi-card {{
      background: {GRAY_100};
      border: 1px solid {GRAY_200};
      border-radius: 10px;
      padding: 20px 18px 18px 18px;
  }}
  .kpi-label {{
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {GRAY_500};
      margin-bottom: 8px;
  }}
  .kpi-value {{
      font-size: 1.8rem;
      font-weight: 800;
      color: {GRAY_900};
      line-height: 1;
  }}
  .kpi-sub {{
      font-size: 0.75rem;
      color: {GRAY_500};
      margin-top: 5px;
  }}
  .kpi-card-alert {{
      background: {RED_LIGHT};
      border: 1px solid #FECACA;
      border-radius: 10px;
      padding: 20px 18px 18px 18px;
  }}
  .kpi-value-alert {{
      font-size: 1.8rem;
      font-weight: 800;
      color: {RED};
      line-height: 1;
  }}

  /* ── Chart card wrapper ── */
  .chart-card {{
      background: {WHITE};
      border: 1px solid {GRAY_200};
      border-radius: 10px;
      padding: 4px;
  }}

  /* ── Product table segment label ── */
  .seg-badge {{
      display: inline-block;
      background: {BLUE_LIGHT};
      color: {BLUE};
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 3px 10px;
      border-radius: 20px;
      margin-bottom: 8px;
  }}
  .seg-badge-risk {{
      background: {RED_LIGHT};
      color: {RED};
  }}

  /* ── Customer profile card ── */
  .profile-header {{
      font-size: 1rem;
      font-weight: 700;
      color: {GRAY_900};
      margin: 1rem 0 0.8rem 0;
  }}
  .rec-card {{
      background: {BLUE_LIGHT};
      border-left: 4px solid {BLUE};
      border-radius: 6px;
      padding: 14px 18px;
      margin-top: 14px;
      font-size: 0.88rem;
      color: {GRAY_700};
  }}
  .rec-card strong {{ color: {GRAY_900}; }}

  /* ── Upload area ── */
  [data-testid="stFileUploader"] {{
      border: 2px dashed {GRAY_200} !important;
      border-radius: 10px !important;
      padding: 8px !important;
  }}

  /* ── Streamlit metric overrides ── */
  [data-testid="stMetricValue"] {{
      font-size: 1.4rem !important;
      font-weight: 700 !important;
      color: {GRAY_900} !important;
  }}
  [data-testid="stMetricLabel"] {{
      font-size: 0.78rem !important;
      color: {GRAY_500} !important;
  }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(number: str, title: str, desc: str = ""):
    desc_html = f'<p class="section-desc">{desc}</p>' if desc else ""
    st.markdown(f"""
    <div class="section-header">
      <div class="section-number">{number}</div>
      <div>
        <p class="section-title">{title}</p>
        {desc_html}
      </div>
    </div>""", unsafe_allow_html=True)


def kpi(col, label, value, sub="", alert=False):
    card_cls  = "kpi-card-alert" if alert else "kpi-card"
    value_cls = "kpi-value-alert" if alert else "kpi-value"
    sub_html  = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(f"""
    <div class="{card_cls}">
      <div class="kpi-label">{label}</div>
      <div class="{value_cls}">{value}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)


def chart_defaults(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=GRAY_900), x=0, xanchor="left"),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(color=GRAY_700, size=11),
        margin=dict(l=16, r=16, t=44, b=16),
        height=height,
        legend=dict(
            bgcolor=WHITE,
            bordercolor=GRAY_200,
            borderwidth=1,
            font=dict(size=11, color=GRAY_700),
        ),
        xaxis=dict(
            showgrid=True, gridcolor=GRAY_200, gridwidth=1,
            linecolor=GRAY_200, tickfont=dict(color=GRAY_700),
            title_font=dict(color=GRAY_700),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRAY_200, gridwidth=1,
            linecolor=GRAY_200, tickfont=dict(color=GRAY_700),
            title_font=dict(color=GRAY_700),
        ),
    )
    return fig


# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes: bytes):
    raw = pd.read_csv(io.BytesIO(file_bytes), encoding="ISO-8859-1")
    df  = clean_data(raw)
    rfm = create_rfm(df)
    rfm = add_additional_features(rfm, df)
    rfm = train_kmeans(rfm, n_clusters=4)
    rfm = label_segments(rfm)
    rfm = calculate_clv(rfm)
    rfm = create_churn_label(rfm)
    rfm = train_churn_model(rfm)
    retention = cohort_analysis(df)
    return rfm, df, retention


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
  <div class="page-eyebrow">Analytics Platform</div>
  <h1 class="page-title">Retail Customer Intelligence Dashboard</h1>
  <p class="page-subtitle">Upload a retail transactions CSV to run the full analytics pipeline.</p>
</div>
""", unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 1rem;color:{GRAY_500};
                border:2px dashed {GRAY_200};border-radius:12px;margin-top:1rem;">
      <div style="font-size:2rem;margin-bottom:10px;">📂</div>
      <div style="font-size:0.95rem;font-weight:600;color:{GRAY_700};">No file uploaded yet</div>
      <div style="font-size:0.82rem;margin-top:4px;">Use the uploader above to load your retail transactions CSV.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
with st.spinner("Running pipeline..."):
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

# ═══════════════════════════════════════════════════════════════════════════════
#  KPI SECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)

total_customers = len(rfm)
total_revenue   = df_clean["TotalPrice"].sum()
avg_clv         = rfm["CLV"].mean()
high_risk       = int((rfm["ChurnProb"] > 0.7).sum())
high_risk_pct   = high_risk / total_customers * 100

c1, c2, c3, c4 = st.columns(4, gap="medium")
kpi(c1, "Total Customers",    f"{total_customers:,}",      sub="Unique customer IDs")
kpi(c2, "Total Revenue",      f"£{total_revenue:,.0f}",    sub="Sum of all transactions")
kpi(c3, "Average CLV",        f"£{avg_clv:,.2f}",          sub="Customer lifetime value")
kpi(c4, "High Risk Customers",f"{high_risk:,}",
    sub=f"{high_risk_pct:.1f}% of customer base", alert=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SEGMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
section("1", "Segment Analysis", "Customer distribution and revenue contribution by segment")

seg_counts = rfm["SegmentLabel"].value_counts().reset_index()
seg_counts.columns = ["Segment", "Customers"]

seg_revenue = (
    rfm.groupby("SegmentLabel")["Monetary"].sum().reset_index()
    .rename(columns={"SegmentLabel": "Segment", "Monetary": "Revenue"})
)

col_a, col_b = st.columns(2, gap="medium")

with col_a:
    fig_bar = px.bar(
        seg_counts.sort_values("Customers", ascending=True),
        x="Customers", y="Segment",
        orientation="h",
        color="Segment",
        color_discrete_map={s: SEG_COLORS.get(s, BLUE) for s in seg_counts["Segment"]},
        text="Customers",
    )
    fig_bar.update_traces(textposition="outside", textfont_size=11, marker_line_width=0)
    fig_bar = chart_defaults(fig_bar, "Customer Segment Distribution")
    fig_bar.update_layout(showlegend=False, xaxis_title="Number of Customers", yaxis_title="")
    fig_bar.update_xaxes(showgrid=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    fig_pie = px.pie(
        seg_revenue, names="Segment", values="Revenue",
        color="Segment",
        color_discrete_map={s: SEG_COLORS.get(s, BLUE) for s in seg_revenue["Segment"]},
        hole=0.42,
    )
    fig_pie.update_traces(
        textposition="outside",
        textinfo="percent+label",
        textfont_size=11,
        marker=dict(line=dict(color=WHITE, width=2)),
    )
    fig_pie = chart_defaults(fig_pie, "Revenue Contribution by Segment")
    fig_pie.update_layout(showlegend=True, legend=dict(orientation="v"))
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CUSTOMER BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════════
section("2", "Customer Behavior", "Purchase frequency vs total spend, coloured by segment")

rfm_plot = rfm.reset_index()
fig_scatter = px.scatter(
    rfm_plot, x="Frequency", y="Monetary",
    color="SegmentLabel",
    color_discrete_map={s: SEG_COLORS.get(s, BLUE) for s in rfm_plot["SegmentLabel"].unique()},
    opacity=0.65,
    labels={"Frequency": "Purchase Frequency", "Monetary": "Total Spend (£)", "SegmentLabel": "Segment"},
    hover_data={"CustomerID": True, "Frequency": True, "Monetary": ":.2f", "SegmentLabel": True},
)
fig_scatter.update_traces(marker=dict(size=6, line=dict(width=0)))
fig_scatter = chart_defaults(fig_scatter, "Frequency vs Monetary Value by Segment", height=420)
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CHURN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
section("3", "Churn Analysis", "Churn risk distribution and CLV exposure by risk level")

col_c, col_d = st.columns(2, gap="medium")

with col_c:
    fig_hist = px.histogram(
        rfm, x="ChurnProb", nbins=30,
        labels={"ChurnProb": "Churn Probability", "count": "Number of Customers"},
        color_discrete_sequence=[BLUE],
    )
    fig_hist.update_traces(marker_line_color=WHITE, marker_line_width=0.8)
    fig_hist.add_vline(x=0.7, line_dash="dash", line_color=RED,
                       annotation_text="High Risk Threshold",
                       annotation_font_color=RED, annotation_font_size=10)
    fig_hist = chart_defaults(fig_hist, "Churn Risk Distribution")
    fig_hist.update_layout(xaxis_title="Churn Probability", yaxis_title="Number of Customers")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_d:
    rfm_risk = rfm.copy()
    rfm_risk["Risk Level"] = rfm_risk["ChurnProb"].apply(
        lambda x: "High Risk (> 0.7)" if x > 0.7 else "Low Risk (≤ 0.7)"
    )
    fig_clv = px.scatter(
        rfm_risk.reset_index(), x="CLV", y="Recency",
        color="Risk Level",
        color_discrete_map={"High Risk (> 0.7)": RED, "Low Risk (≤ 0.7)": BLUE},
        opacity=0.65,
        labels={"CLV": "Customer Lifetime Value (£)", "Recency": "Recency (days)"},
        hover_data={"CustomerID": True, "CLV": ":.2f", "Recency": True},
    )
    fig_clv.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig_clv = chart_defaults(fig_clv, "CLV vs Recency — Risk Level")
    st.plotly_chart(fig_clv, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — RETENTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
section("4", "Retention Analysis", "Monthly cohort retention rates — how many customers return each month")

try:
    ret = retention.copy()
    ret.index   = ret.index.astype(str)
    ret.columns = [int(c) for c in ret.columns]

    # Replace NaN with None so Plotly renders gaps cleanly
    z_vals = np.where(np.isnan(ret.values.astype(float)), None, np.round(ret.values.astype(float) * 100, 1))

    fig_heat = go.Figure(go.Heatmap(
        z=z_vals,
        x=[f"M+{c}" for c in ret.columns],
        y=ret.index.tolist(),
        colorscale=[[0.0, BLUE_LIGHT], [0.5, BLUE_MID], [1.0, BLUE]],
        zmin=0, zmax=100,
        text=[[f"{v}%" if v is not None else "" for v in row] for row in z_vals],
        texttemplate="%{text}",
        textfont=dict(size=10, color=GRAY_900),
        showscale=True,
        colorbar=dict(
            title=dict(text="Retention %", font=dict(size=11, color=GRAY_700)),
            tickfont=dict(size=10, color=GRAY_700),
            ticksuffix="%",
        ),
        hoverongaps=False,
        hovertemplate="Cohort: %{y}<br>Month: %{x}<br>Retention: %{z}%<extra></extra>",
    ))
    fig_heat.update_layout(
        title=dict(text="Monthly Cohort Retention Rate", font=dict(size=13, color=GRAY_900), x=0),
        xaxis=dict(title="Months Since First Purchase", tickfont=dict(color=GRAY_700),
                   title_font=dict(color=GRAY_700), side="bottom"),
        yaxis=dict(title="Cohort Month", tickfont=dict(color=GRAY_700),
                   title_font=dict(color=GRAY_700), autorange="reversed"),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(color=GRAY_700),
        margin=dict(l=16, r=16, t=44, b=16),
        height=520,
    )
    st.plotly_chart(fig_heat, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render cohort heatmap: {e}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PRODUCT INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
section("5", "Product Insights", "Top 5 products purchased by customers in each segment")

try:
    top_products = get_top_products_per_cluster(df_clean, rfm, top_n=5)
    segments_sorted = sorted(top_products["SegmentLabel"].unique())
    prod_cols = st.columns(len(segments_sorted), gap="medium")

    for i, seg in enumerate(segments_sorted):
        with prod_cols[i]:
            badge_cls = "seg-badge-risk" if seg == "At Risk" else "seg-badge"
            st.markdown(f'<div class="{badge_cls}">{seg}</div>', unsafe_allow_html=True)
            seg_df = (
                top_products[top_products["SegmentLabel"] == seg][["Description", "TotalQuantity"]]
                .reset_index(drop=True)
                .rename(columns={"Description": "Product", "TotalQuantity": "Qty Sold"})
            )
            seg_df.index += 1
            st.dataframe(
                seg_df,
                use_container_width=True,
                hide_index=False,
                column_config={
                    "Product":  st.column_config.TextColumn("Product", width="large"),
                    "Qty Sold": st.column_config.NumberColumn("Qty Sold", format="%d"),
                },
            )
except Exception as e:
    st.warning(f"Could not load product insights: {e}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CUSTOMER LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
section("6", "Customer Lookup", "Search any customer by ID to view their profile and recommendation")

RECOMMENDATIONS = {
    "Champions":          ("🏆", "Reward them",    "Offer loyalty perks or early access to new products."),
    "Loyal Customers":    ("⬆️",  "Upsell",         "Suggest premium products or bundle deals."),
    "Potential Loyalists":("🌱", "Nurture",         "Send personalised offers to increase engagement."),
    "At Risk":            ("🔔", "Re-engage",       "Send a win-back campaign with a discount or reminder."),
}

rfm_indexed = rfm.reset_index()

search_col, _ = st.columns([1, 2])
with search_col:
    customer_id = st.number_input(
        "Customer ID", min_value=0, step=1, value=0,
        label_visibility="visible",
        help="Enter a CustomerID from the dataset",
    )

if customer_id != 0:
    match = rfm_indexed[rfm_indexed["CustomerID"] == int(customer_id)]

    if match.empty:
        st.error(f"Customer ID {int(customer_id)} was not found in the dataset.")
    else:
        row = match.iloc[0]
        seg = row["SegmentLabel"]

        st.markdown(
            f'<div class="profile-header">Profile — Customer #{int(customer_id)}'
            f'&nbsp;&nbsp;<span style="font-size:0.75rem;font-weight:600;'
            f'background:{BLUE_LIGHT};color:{BLUE};padding:3px 10px;'
            f'border-radius:20px;">{seg}</span></div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4, m5 = st.columns(5, gap="medium")
        m1.metric("Recency",           f"{int(row['Recency'])} days")
        m2.metric("Frequency",         f"{int(row['Frequency'])} orders")
        m3.metric("Monetary",          f"£{row['Monetary']:,.2f}")
        m4.metric("CLV",               f"£{row['CLV']:,.2f}")
        m5.metric("Churn Probability", f"{row['ChurnProb']:.1%}")

        if seg in RECOMMENDATIONS:
            icon, action, detail = RECOMMENDATIONS[seg]
            st.markdown(
                f'<div class="rec-card">'
                f'{icon} <strong>{action}:</strong> {detail}'
                f'</div>',
                unsafe_allow_html=True,
            )
