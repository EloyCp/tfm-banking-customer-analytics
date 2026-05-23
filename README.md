# Banking Customer Analytics — Master's Thesis (TFM)

> **Master's Final Project (Trabajo Fin de Máster)** — Nuclio Digital School · Data Science & AI

An end-to-end Data Science project for a retail bank: from raw transactional data to an actionable cross-sell campaign with projected revenue. The pipeline combines **unsupervised segmentation** (customer clustering) and **supervised modeling** (per-product propensity), and is structured as three sequential scripts that mirror a real production workflow.

The business question: *"Which products should we promote, to which customers, and what revenue can we expect?"*

---

## Project Overview

The dataset comes from a Spanish retail bank's monthly customer snapshots between January 2018 and May 2019, covering ~2 million rows over 13 months. Each row is a (customer, month) pair with sociodemographic information, commercial activity, product holdings and transactional sales.

The project answers three layered questions:

1. **Who are our customers?** → Customer segmentation using K-Means clustering.
2. **Who is most likely to buy each product next month?** → Per-product propensity modeling.
3. **What is the expected revenue of the next cross-sell campaign?** → Top-5% scoring + historical conversion rate + mean margin per unit.

---

## Pipeline Architecture

The repository is structured as **three sequential scripts** that pass intermediate parquet files between them. This mirrors how a productionized ML pipeline would be organized.

```
data/                          outputs/
  customer_new.parquet  ─┐       df_final_preparado.parquet  ─┐
  sales.csv             ─┼─►  01 ──►                          ├─►  03 ──►  campaign + revenue projection
  product_description ──┘                                     │
                              df_cluster.parquet  ─────────── 02 ──┘
```

### Script 1 — `01_data_preparation.py`
Cleans, merges and feature-engineers the raw datasets into a single analytical table.
- Data Understanding diagnostics (shapes, types, nulls, duplicates, value counts).
- **Family-aware null imputation**: `Other` placeholder for categoricals, mode for booleans, median for skewed numericals.
- Merges sales (aggregated per month) onto the customer table to produce a `net_margin` target per (customer, month).
- Drops deceased customers and zero-variance / single-market columns (`em_account_pp`, `country_id`, ...).
- Automatic column classification → dates / booleans / numerical / categorical.
- **Outlier capping** before imputation (salary at p95, age clipped to [18, 90]).
- **Top-7 + 'Otros' grouping** on categoricals followed by int8 One-Hot Encoding (memory-aware).
- Boolean homogenization (gender mapped, products cast to int8).

### Script 2 — `02_clustering_model.py`
Segments active customers into business-meaningful groups using **MiniBatchKMeans** (chosen over standard KMeans for scalability on 2M rows).
- Snapshot filter: May 2019 + `active_customer = 1` (otherwise the algorithm just splits active vs. inactive).
- Elbow method on a 9-K range to validate the chosen K.
- Two feature-set experiments (initial vs. extended) to verify segment **stability**.
- Cross-validation against traditional KMeans as a sanity check.
- Final **5 business segments**:

| Cluster | Business name | Key signal |
|---------|---------------|------------|
| 0 | Mature account + EMC customers | Heavy `em_acount`/`emc_account` use, low margin |
| 1 | Young customers with basic checking account | Low age, low margin, basic product |
| 2 | High-net-worth investors | High salary, 100% with funds + securities |
| 3 | High-bonding payroll customers | Top margin, payroll + pension + debit |
| 4 | High-salary low-bonding customers | High salary but few products → untapped potential |

### Script 3 — `03_propensity_model.py`
Builds per-product propensity models and projects the next campaign.
- **Product profitability analysis** → top 4 products concentrate 93.7% of total margin; `pension_plan` alone is 79%.
- **Target engineering**: predicts the *acquisition event* (0 → 1 state change), not the static "owns product X" flag.
- **Feature engineering**: lag-1 versions of every input variable to prevent target leakage.
- **Strict temporal split**: Apr 2019 = Test, May 2019 = Validation, everything before = Train.
- For each candidate product the pipeline:
  - selects the top 7 most important features with a Random Forest,
  - trains a balanced Decision Tree on those features,
  - reports classification report + confusion matrix + ROC curve,
  - computes **SHAP values** to explain the predictions,
  - simulates the **real conversion rate** on the validation month.
- Drops `short_term_deposit` (too few positives to evaluate) and `long_term_deposit` (very low real conversion).
- Final campaign products: **pension_plan** (primary) + **em_acount** (secondary).
- **June 2019 campaign projection**: top 5% scoring → expected acquisitions → expected revenue.
- Cross-references the top pension_plan candidates with the customer segments from Script 2.

---

## Key Design Decisions

A few choices that distinguish this work from a textbook pipeline:

- **Real *acquisition* targets, not static ownership flags.** A model that learns "this client owns pension plan" is useless for a campaign — the bank already knows that. We compute the state change `0 → 1` so the model learns to predict *new* contracts, not existing ones.
- **Lag-1 features for strict temporal honesty.** Every input variable is shifted by one month, so the model never sees information from the target month itself.
- **Real conversion rate over the validation month, not just AUC.** AUC tells you about ranking quality; the marketing team cares about *"if I call the top 5%, how many will actually buy?"*. We compute that figure explicitly.
- **Family-aware null imputation.** Instead of applying one strategy across the board, each variable family (high-cardinality categorical, boolean, skewed numerical) gets the treatment that makes sense for it.
- **MiniBatchKMeans over KMeans.** With 2M rows full KMeans is expensive; MiniBatch is much faster and we explicitly benchmark both to confirm the segmentation is consistent.
- **SHAP for model explainability.** The Decision Tree's feature importances are aggregate; SHAP reveals exactly how each variable pushed each prediction up or down, which is what a banker actually wants to see.

---

## Business Results

| Output | Value |
|--------|-------|
| Final analytical features | 92 (from ~25 raw columns) |
| Customer segments identified | 5 actionable business segments |
| Products selected for campaign | 2 (pension_plan + em_acount) |
| Models built | 4 (one per candidate product) |
| Top-5% real conversion rate (pension_plan) | Real conversion measured on validation |
| Output | Ranked client lists + revenue projection for June 2019 |

The model translates a tangled customer database into a **prioritized, segmented contact list with an expected euro value** — exactly the type of output a marketing team can act on.

---

## Technologies Used

- **Python 3.10+**
- **pandas** & **NumPy** — data manipulation (millions of rows)
- **pyarrow** — parquet I/O (intermediate persistence)
- **Matplotlib** & **seaborn** — visualizations
- **scikit-learn** — `StandardScaler`, `MiniBatchKMeans`, `KMeans`, `RandomForestClassifier`, `DecisionTreeClassifier`, metrics
- **SHAP** — model explainability

---

## Repository Structure

```
tfm-banking-customer-analytics/
├── 01_data_preparation.py     # Script 1 — cleaning + feature engineering
├── 02_clustering_model.py     # Script 2 — customer segmentation
├── 03_propensity_model.py     # Script 3 — propensity + campaign projection
├── data/
│   ├── customer_new.parquet   # Merged customer table (~8.5 MB)
│   ├── sales.csv              # Sales transactions
│   └── product_description.csv  # Product catalog
├── outputs/                   # Intermediate artifacts (generated by the scripts)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Files excluded from version control
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/EloyCp/tfm-banking-customer-analytics.git
cd tfm-banking-customer-analytics
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the scripts **in order**

```bash
python 01_data_preparation.py   # ≈ 1–2 minutes, produces outputs/df_final_preparado.parquet
python 02_clustering_model.py   # ≈ 1 minute,    produces outputs/df_cluster.parquet
python 03_propensity_model.py   # ≈ 3–5 minutes, console reports + plots
```

Each script logs every diagnostic step to the console and renders a series of plots (target distribution, elbow curves, feature importances, ROC curves, confusion matrices, SHAP summaries). Output files land in `outputs/`.

> **Heads-up:** the full pipeline involves 4 Decision Tree models, 4 Random Forest feature selectors and 4 SHAP analyses. On a modern laptop the longest stretch is Script 3 (about 3–5 minutes).

---

## License

This project was developed for educational purposes as the Master's Final Project (TFM) at Nuclio Digital School.
