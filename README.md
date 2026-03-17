# Retail Customer Intelligence System

## Overview

A production-style Customer Intelligence Platform built on the Online Retail
dataset. It converts a data science notebook into a fully modular ML pipeline
with an interactive Streamlit dashboard.

## Features

| Feature | Description |
|---|---|
| RFM Segmentation | K-Means clusters customers into Champions, Loyal Customers, Potential Loyalists, At Risk |
| Churn Prediction | RandomForest model outputs churn probability per customer |
| CLV Calculation | CLV = (Monetary x Frequency) / (Recency + 1) |
| Cohort Retention | Monthly cohort heatmap showing retention over time |
| Product Recommendations | Top 5 products per customer segment |
| Customer Search | Full profile lookup by CustomerID with recommended action |

## Pipeline

```
load_data + clean_data
        |
   create_rfm + add_additional_features
        |
   train_kmeans + label_segments
        |
   calculate_clv
        |
   create_churn_label + train_churn_model
        |
   get_top_products_per_cluster + cohort_analysis
        |
   Streamlit Dashboard
```

## Dataset

Place the file at: `data/Online Retail.csv`

UK-based online retailer transactions (2010-2011).
Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
UnitPrice, CustomerID, Country.

## Project Structure

```
retail-customer-intelligence/
  data/
    Online Retail.csv
  src/
    preprocessing.py        load_data, clean_data
    feature_engineering.py  create_rfm, add_additional_features
    segmentation_model.py   train_kmeans, label_segments
    analytics.py            calculate_clv, cluster_summary, cohort_analysis
    churn_model.py          create_churn_label, train_churn_model
    recommendation.py       get_top_products_per_cluster
  model/
    kmeans_model.pkl
    scaler.pkl
    churn_model.pkl
  dashboard/
    streamlit_app.py
  notebooks/
    Retail_Customer_Segmentation.ipynb
  requirements.txt
  README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run dashboard/streamlit_app.py
```
