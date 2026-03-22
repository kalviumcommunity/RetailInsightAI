# Retail Customer Intelligence System

A production-style machine learning pipeline and interactive Streamlit dashboard built on the [Online Retail dataset](https://archive.ics.uci.edu/ml/datasets/online+retail). Transforms raw transaction data into actionable customer intelligence — segmentation, churn prediction, CLV, cohort retention, and product recommendations.

---

## Features

| Module | What it does |
|---|---|
| RFM Segmentation | K-Means clustering assigns customers to Champions, Loyal Customers, Potential Loyalists, or At Risk |
| Churn Prediction | Random Forest model outputs a churn probability score per customer |
| CLV Calculation | `CLV = (Monetary × Frequency) / (Recency + 1)` |
| Cohort Retention | Monthly cohort heatmap showing how many customers return over time |
| Product Recommendations | Top 5 products per segment based on purchase volume |
| Customer Lookup | Full profile by CustomerID with a segment-specific recommended action |

---

## Project Structure

```
retail-customer-intelligence/
├── data/
│   └── Online Retail.csv          # Raw transaction dataset
├── src/
│   ├── preprocessing.py           # load_data, clean_data
│   ├── feature_engineering.py     # create_rfm, add_additional_features
│   ├── segmentation_model.py      # train_kmeans, label_segments
│   ├── analytics.py               # calculate_clv, cluster_summary, cohort_analysis
│   ├── churn_model.py             # create_churn_label, train_churn_model
│   └── recommendation.py         # get_top_products_per_cluster
├── model/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── churn_model.pkl
├── dashboard/
│   └── streamlit_app.py           # Streamlit UI
├── notebooks/
│   └── Retail_Customer_Segmentation.ipynb
├── requirements.txt
└── README.md
```

---

## Pipeline

```
load_data → clean_data
    ↓
create_rfm → add_additional_features
    ↓
train_kmeans → label_segments
    ↓
calculate_clv
    ↓
create_churn_label → train_churn_model
    ↓
cohort_analysis + get_top_products_per_cluster
    ↓
Streamlit Dashboard
```

---

## Dataset

Place the file at `data/Online Retail.csv`.

UK-based online retailer, transactions from 2010–2011.

| Column | Description |
|---|---|
| InvoiceNo | Unique invoice number |
| StockCode | Product code |
| Description | Product name |
| Quantity | Units purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit (£) |
| CustomerID | Unique customer identifier |
| Country | Customer country |

---

## Setup

**1. Clone the repo and install dependencies**

```bash
pip install -r requirements.txt
```

**2. Add the dataset**

```
data/Online Retail.csv
```

**3. Run the dashboard**

```bash
streamlit run dashboard/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser, upload the CSV, and the full pipeline runs automatically.

---

## Dashboard Sections

| # | Section | Description |
|---|---|---|
| — | KPI Bar | Total customers, revenue, average CLV, high-risk count |
| 1 | Segment Analysis | Customer count and revenue share per segment |
| 2 | Customer Behavior | Frequency vs monetary scatter by segment |
| 3 | Churn Analysis | Churn probability histogram + CLV vs recency risk plot |
| 4 | Retention Analysis | Monthly cohort retention heatmap |
| 5 | Product Insights | Top 5 products per segment |
| 6 | Customer Lookup | Profile + churn score + recommended action by CustomerID |

---

## Tech Stack

| Layer | Library |
|---|---|
| Data | pandas, numpy |
| ML | scikit-learn |
| Visualisation | plotly |
| Dashboard | streamlit |
| Model persistence | joblib |
| Excel support | openpyxl |
