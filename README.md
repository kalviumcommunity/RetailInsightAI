# Retail Customer Segmentation System

## Project Overview

Analyses customer purchase behaviour from the Online Retail dataset,
segments customers using RFM + K-Means clustering, calculates Customer
Lifetime Value, and surfaces targeted marketing recommendations through
an interactive Streamlit dashboard.

## Dataset

Place the file at: `data/Online Retail.csv`

The dataset contains transactional records from a UK-based online retailer
(2010-2011). Key columns: InvoiceNo, StockCode, Description, Quantity,
InvoiceDate, UnitPrice, CustomerID, Country.

## RFM Segmentation

Each customer is scored on three dimensions:
- Recency  — days since last purchase (lower = more recent)
- Frequency — number of unique invoices
- Monetary  — total spend

## How Clustering Works

1. Raw transactions are cleaned (nulls, returns, zero-price rows removed).
2. RFM features are computed per customer.
3. Features are scaled with StandardScaler.
4. K-Means (k=4) assigns each customer to a cluster.
5. CLV is derived as: CLV = (Monetary x Frequency) / (Recency + 1)

## Project Structure

```
project/
  data/           Online Retail.csv
  src/            preprocessing.py  clustering.py  analytics.py  recommendation.py
  model/          kmeans_model.pkl  scaler.pkl  (auto-generated on first run)
  app/            streamlit_app.py
  notebooks/      Retail_Customer_Segmentation.ipynb
  requirements.txt
  README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```
