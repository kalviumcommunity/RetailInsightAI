# Setup & Getting Started — Retail Customer Intelligence System

This document covers everything you need to get the project running locally from scratch.

---

## Prerequisites

Make sure you have the following installed before starting:

| Requirement | Version | Check |
|---|---|---|
| Python | 3.9 or higher | `python --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-org/retail-customer-intelligence.git
cd retail-customer-intelligence
```

---

## 2. Create a Virtual Environment

It is strongly recommended to use a virtual environment to keep dependencies isolated.

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt once activated.

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Library | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| scikit-learn | KMeans clustering, Random Forest churn model |
| plotly | Interactive charts in the dashboard |
| streamlit | Dashboard framework |
| joblib | Saving and loading model `.pkl` files |
| matplotlib | Supporting visualisations |
| seaborn | Supporting visualisations |
| openpyxl | Excel file support |

---

## 4. Add the Dataset

Download the **Online Retail** dataset and place it at:

```
data/Online Retail.csv
```

The file must be named exactly `Online Retail.csv` (with a space) and encoded in `ISO-8859-1`.

> Dataset source: [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail)

Expected columns:

```
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
```

---

## 5. Project Structure

Before running anything, your folder should look like this:

```
retail-customer-intelligence/
├── data/
│   └── Online Retail.csv          ← place dataset here
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── segmentation_model.py
│   ├── analytics.py
│   ├── churn_model.py
│   └── recommendation.py
├── model/                         ← .pkl files are saved here automatically
├── dashboard/
│   └── streamlit_app.py
├── notebooks/
│   └── Retail_Customer_Segmentation.ipynb
├── requirements.txt
├── README.md
├── TEAM_PLAN.md
└── SETUP.md
```

---

## 6. Run the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 7. Using the Dashboard

Once the app loads in your browser:

1. Click **Browse files** and upload `data/Online Retail.csv`
2. The full pipeline runs automatically in this order:
   - Data cleaning
   - RFM feature engineering
   - KMeans segmentation + segment labelling
   - CLV calculation
   - Churn label creation + Random Forest training
   - Cohort analysis + product recommendations
3. Model files are saved to `model/` on first run and reused on subsequent runs
4. All 6 dashboard sections populate with charts and tables
5. Use the **Customer Lookup** section at the bottom to search any CustomerID

> The first run takes longer because it trains the models. Subsequent runs use Streamlit's `st.cache_data` and are much faster.

---

## 8. Running the Notebook (Optional)

To explore the data and pipeline interactively:

```bash
pip install jupyter
jupyter notebook notebooks/Retail_Customer_Segmentation.ipynb
```

---

## Common Issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'plotly'` | Run `pip install plotly` |
| `ModuleNotFoundError: No module named 'src'` | Make sure you run `streamlit run` from the project root, not from inside `dashboard/` |
| Dataset not found / encoding error | Confirm the file is at `data/Online Retail.csv` and is CSV encoded in ISO-8859-1 |
| `model/` files missing on first run | This is expected — models are trained and saved automatically on first upload |
| Port 8501 already in use | Run `streamlit run dashboard/streamlit_app.py --server.port 8502` |

---

## Scope of Contribution (SoC)

| Person | Track | Files Owned |
|---|---|---|
| **Heramb** | Data Engineering & ML Pipeline | `src/preprocessing.py`, `src/feature_engineering.py`, `src/segmentation_model.py`, `model/kmeans_model.pkl`, `model/scaler.pkl` |
| **Shivam** | Analytics & Churn Modelling | `src/analytics.py`, `src/churn_model.py`, `src/recommendation.py`, `model/churn_model.pkl`, `notebooks/` |
| **Aayush** | Streamlit Dashboard & UI | `dashboard/streamlit_app.py`, `requirements.txt` |

Each person is responsible for testing their own modules before integration. See `TEAM_PLAN.md` for the full day-by-day breakdown.
