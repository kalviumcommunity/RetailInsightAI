# Team Task Division — Retail Customer Intelligence System

**Timeline:** 17 Days  
**Team:** Heramb, Shivam, Aayush

---

## Track Overview

| Person | Track | Owns |
|---|---|---|
| Heramb | Data Engineering & ML Pipeline | preprocessing, feature engineering, segmentation, model artefacts |
| Shivam | Analytics & Churn Modelling | CLV, cohort analysis, churn model, recommendations |
| Aayush | Streamlit Dashboard & UI | full dashboard, charts, layout, interactivity |

---

## Day-by-Day Plan

| Day | Heramb | Shivam | Aayush |
|---|---|---|---|
| 1 | Understand dataset, validate columns, document edge cases (nulls, cancellations, negatives) | Study the notebook, understand existing EDA, identify analytics needed | Set up Streamlit app skeleton — page config, global CSS, header |
| 2 | Continue dataset validation, write data quality notes | Map out CLV formula and cohort logic on paper | Build file uploader, pipeline call with `st.cache_data`, error handling |
| 3 | Implement `load_data` and `clean_data` in `preprocessing.py` | Implement and test `calculate_clv` — verify formula `(Monetary × Frequency) / (Recency + 1)` | Build KPI section — 4-column layout with custom HTML cards |
| 4 | Test `clean_data` — confirm nulls removed, negatives dropped, `TotalPrice` computed correctly | Test CLV output against manual calculations | Polish KPI cards — labels, formatting, high-risk alert styling |
| 5 | Implement `create_rfm` in `feature_engineering.py` | Implement `cohort_analysis` — build monthly cohort pivot table | Build Section 1 bar chart — Customer Segment Distribution (Plotly horizontal bar) |
| 6 | Implement `add_additional_features` — `AvgOrderValue`, `PurchaseInterval` | Test cohort retention rates, validate pivot output | Build Section 1 pie chart — Revenue Contribution by Segment (donut) |
| 7 | Implement `train_kmeans` in `segmentation_model.py`, save `kmeans_model.pkl` and `scaler.pkl` | Define churn label — `Churned = 1` if `Recency > 90`, implement `create_churn_label` | Build Section 2 — Frequency vs Monetary scatter plot coloured by segment |
| 8 | Implement `label_segments` — score clusters by composite RFM rank, assign human-readable labels | Train Random Forest in `train_churn_model`, attach `ChurnProb` column | Build Section 3 — Churn probability histogram with high-risk threshold line |
| 9 | Run end-to-end test: raw CSV → labelled RFM DataFrame with `SegmentLabel` | Save `churn_model.pkl`, test model loads and predicts correctly | Build Section 3 — CLV vs Recency scatter coloured by risk level |
| 10 | Fix any bugs found in pipeline, verify all column names match spec | Implement `get_top_products_per_cluster` in `recommendation.py` | Build Section 4 — Cohort retention heatmap (Plotly, blue gradient, % annotations) |
| 11 | Run full pipeline with Shivam's outputs, confirm DataFrame shape is correct end-to-end | Test top products output per segment, verify quantities are correct | Build Section 5 — Product insights tables with segment badge labels |
| 12 | Write inline docstrings for all functions in `preprocessing.py` and `feature_engineering.py` | Implement `cluster_summary` in `analytics.py` — mean RFM/CLV/ChurnProb per segment | Build Section 6 — Customer lookup: number input, 5-column metric row |
| 13 | Write inline docstrings for `segmentation_model.py`, clean up code | Write docstrings for `churn_model.py` and `recommendation.py` | Build Section 6 — recommendation card with icon and action text |
| 14 | Support Aayush — fix any data shape or column name issues found during dashboard integration | Integration test — run full pipeline with Heramb's output, confirm all columns present | Full end-to-end test with real CSV — all 6 sections rendering with real data |
| 15 | Verify models save and load correctly from `model/` directory | Tune churn threshold if needed based on distribution, final model check | Full UI pass — spacing, dividers, font consistency, colour palette audit |
| 16 | Final code review of all `src/` modules, ensure no hardcoded paths | Final review of analytics outputs, cross-check numbers with notebook | Fix any rendering bugs found in day 15 testing, final polish |
| 17 | Project complete — confirm pipeline runs clean from scratch on a fresh CSV | Project complete — confirm all analytical outputs are correct | Project complete — confirm dashboard runs without errors, ready to present |

---
