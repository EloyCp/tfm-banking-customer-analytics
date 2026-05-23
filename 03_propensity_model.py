"""
TFM — Banking Customer Analytics
=================================
Script 3 of 3 — Propensity Modeling for Cross-sell Campaigns

Loads the prepared dataset and the sales table, ranks the bank's products
by profitability, picks the most relevant ones as campaign targets and
builds a per-product propensity model that scores every eligible client
based on their probability of contracting that product next month.

Modeling notes:
    - Targets are NOT the static "client owns product X" flag. They are the
      state change "client did NOT own X in T-1 and DOES own X in T" — i.e.
      a real *new acquisition* event. This is what the marketing team wants
      to predict.
    - Features are LAG-1 versions of every input variable so the model only
      sees information that was available before the target month (no
      target leakage).
    - The temporal split is strict: Apr 2019 = Test, May 2019 = Validation,
      everything before = Train.
    - For each target the pipeline:
        (a) selects the top 7 most important features with a Random Forest,
        (b) trains a balanced Decision Tree on those features,
        (c) reports a classification report, confusion matrix and ROC curve,
        (d) computes SHAP values to explain the predictions,
        (e) simulates the real conversion rate of the top 5% of scored
            clients — the only number the business actually cares about.

After scoring, low-conversion products are dropped from the campaign list
(`short_term_deposit`, `long_term_deposit`). The kept ones are:
    - pension_plan (primary target, 79% of total margin).
    - em_acount    (secondary, value-add cross-sell).

The script ends by projecting expected revenue for the June 2019 campaign
based on the predicted top 5% per product and the historical mean margin
per unit, and crosses the top pension_plan candidates with the customer
segments produced by Script 2.

Inputs:
    outputs/df_final_preparado.parquet   (Script 1)
    outputs/df_cluster.parquet           (Script 2)
    data/sales.csv
    data/product_description.csv

Outputs:
    Console reports + plots (no persistent artifact required).
"""

# =============================================================================
# IMPORTS
# =============================================================================
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, roc_curve, auc, confusion_matrix,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_PREPARED = "outputs/df_final_preparado.parquet"
INPUT_CLUSTERS = "outputs/df_cluster.parquet"
INPUT_SALES = "data/sales.csv"
INPUT_PRODUCTS = "data/product_description.csv"


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("Loading prepared dataset...")
df_model = pd.read_parquet(INPUT_PREPARED, engine='pyarrow')
print(df_model.head())
print(df_model.info())

# Sales and product catalog are reloaded because they were dropped at the
# end of Script 1 — we need them again to compute product profitability.
df_sales = pd.read_csv(INPUT_SALES)
df_product_desc = pd.read_csv(INPUT_PRODUCTS)

# Harmonize key names with the rest of the pipeline
if 'cid' in df_sales.columns:
    df_sales = df_sales.rename(columns={'cid': 'pk_cid'})
if 'product_ID' in df_sales.columns:
    df_sales = df_sales.rename(columns={'product_ID': 'pk_product_ID'})


# =============================================================================
# 2. PRODUCT PROFITABILITY ANALYSIS
# =============================================================================
# The goal here is to focus modeling effort on the products that actually
# matter for the bank's bottom line. We aggregate sales per product,
# attach product names, and compute the % contribution to the total margin.
df_product_analysis = df_sales.groupby('pk_product_ID').agg(
    total_sales=('pk_cid', 'count'),
    total_margin=('net_margin', 'sum'),
    mean_margin=('net_margin', 'mean'),
).reset_index()

df_product_analysis = pd.merge(
    df_product_analysis, df_product_desc, on='pk_product_ID', how='left',
)
total_profit = df_product_analysis['total_margin'].sum()
df_product_analysis['pct_margin'] = (
    df_product_analysis['total_margin'] / total_profit * 100
)
df_product_analysis = df_product_analysis.sort_values(
    by='total_margin', ascending=False,
)

print("--- TOP PRODUCTS BY PROFITABILITY ---")
print(df_product_analysis[
    ['product_desc', 'total_sales', 'total_margin', 'pct_margin', 'mean_margin']
].head(10))

# Store the mean margin per unit for the top 4 products. We will reuse this
# downstream when projecting expected campaign revenue.
top4_margins = df_product_analysis.head(4)[
    ['pk_product_ID', 'product_desc', 'mean_margin']
]


# --- Top 7 profitability chart -----------------------------------------------
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=df_product_analysis.head(7),
    x='product_desc', y='total_margin', palette='magma',
)
plt.title('Top 7 Most Profitable Products (Total Net Margin)', fontsize=15)
plt.xticks(rotation=45)
plt.ylabel('Euros (€)')
for i, p in enumerate(ax.patches):
    ax.annotate(
        f"Sales: {df_product_analysis.iloc[i]['total_sales']}",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center', xytext=(0, 9),
        textcoords='offset points', fontsize=10, fontweight='bold',
    )
plt.tight_layout()
plt.show()

# The top 4 products (pension_plan, em_acount, short_term_deposit,
# long_term_deposit) concentrate ~93.7% of the total margin. pension_plan
# alone is 79% → our primary target.

del df_product_analysis
gc.collect()


# =============================================================================
# 3. TARGET ENGINEERING — state change detection (0 → 1)
# =============================================================================
# A meaningful target for cross-sell is the *acquisition* event: the client
# did not own a product last month and owns it this month. We compute
# diff() on the monthly product flag, ordered by client + date, and keep
# only the +1 transitions (new acquisitions).
target_products = [
    'pension_plan_b', 'em_acount_b',
    'short_term_deposit_b', 'long_term_deposit_b',
]

df_model = df_model.sort_values(
    by=['pk_cid_n', 'pk_partition_year', 'pk_partition_month'],
)

print("\nComputing product acquisition targets...")
for prod in target_products:
    df_model[f'target_{prod}'] = df_model.groupby('pk_cid_n')[prod].diff()
    df_model[f'target_{prod}'] = df_model[f'target_{prod}'].apply(
        lambda x: 1 if x == 1 else 0
    )
    total_new = df_model[f'target_{prod}'].sum()
    print(f"  Total new acquisitions for {prod}: {total_new}")


# =============================================================================
# 4. FEATURE ENGINEERING — Lag-1 features
# =============================================================================
# To predict month T we use information from month T-1. Per client we shift
# every base feature by one month. Rows where the lag is undefined (a
# client's first appearance) are dropped.
features_base = [
    # Demographics & commercial activity
    'age_n', 'salary_n', 'gender_b',
    'segment_cat_02 - PARTICULARES', 'segment_cat_03 - UNIVERSITARIO',
    'active_customer_b', 'segment_cat_Other', 'segment_cat_01 - TOP',
    'entry_channel_cat_KAT', 'entry_channel_cat_KFC',
    'entry_channel_cat_KHE', 'entry_channel_cat_KHK',
    'entry_channel_cat_KHQ', 'entry_channel_cat_Otros',
    'region_code_cat_8.0', 'region_code_cat_15.0', 'region_code_cat_28.0',
    'region_code_cat_29.0', 'region_code_cat_30.0', 'region_code_cat_41.0',
    'region_code_cat_46.0', 'region_code_cat_Otros',
    # Product holdings
    'short_term_deposit_b', 'loans_b', 'mortgage_b', 'funds_b',
    'securities_b', 'long_term_deposit_b', 'credit_card_b', 'payroll_b',
    'pension_plan_b', 'payroll_account_b', 'emc_account_b', 'debit_card_b',
    'em_account_p_b', 'em_acount_b',
]
# Keep only the features that actually exist in this dataset
features_base = [f for f in features_base if f in df_model.columns]

for col in features_base:
    df_model[f'{col}_lag1'] = df_model.groupby('pk_cid_n')[col].shift(1)

# Drop the first month per client (no lag information available)
df_ml = df_model.dropna(subset=[f'{features_base[0]}_lag1']).copy()

del df_model
gc.collect()


# =============================================================================
# 5. TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
# Apr 2019 = Test, May 2019 = Validation, everything before = Train.
# Strictly chronological so we never train on the future.
train_data = df_ml[
    ((df_ml['pk_partition_month'] < 4) & (df_ml['pk_partition_year'] == 2019)) |
    (df_ml['pk_partition_year'] == 2018)
]
test_data = df_ml[
    (df_ml['pk_partition_month'] == 4) & (df_ml['pk_partition_year'] == 2019)
]
val_data = df_ml[
    (df_ml['pk_partition_month'] == 5) & (df_ml['pk_partition_year'] == 2019)
]

# Keep only the columns each model actually needs (ID, targets and lags).
target_cols = [f'target_{p}' for p in target_products]
lag_cols = [c for c in df_ml.columns if c.endswith('_lag1')]
keep_cols = ['pk_cid_n'] + target_cols + lag_cols

train_clean = train_data[keep_cols].copy()
test_clean = test_data[keep_cols].copy()
val_clean = val_data[keep_cols].copy()

del train_data, test_data, val_data
gc.collect()

print(f"Train: {train_clean.shape}")
print(f"Test:  {test_clean.shape}")
print(f"Val:   {val_clean.shape}")


# =============================================================================
# 6. PER-PRODUCT PROPENSITY MODELING
# =============================================================================
# Loop over each candidate product and run the full modeling pipeline:
#   (a) restrict the universe to *eligible* customers (don't already own X),
#   (b) Random Forest feature selection → top 7,
#   (c) Decision Tree training (balanced class weight to handle the rare
#       positive class),
#   (d) classification report + confusion matrix + ROC curve,
#   (e) SHAP feature impact analysis,
#   (f) top 5% scoring + real conversion rate on the validation month.

model_results = {}

for prod in target_products:
    target_col = f'target_{prod}'
    product_lag = f'{prod}_lag1'

    print(f"\n{'=' * 50}")
    print(f" Training and evaluating model for: {prod}")
    print('=' * 50)

    # --- (a) Eligibility filter — only customers who don't already own it
    train_universe = train_clean[train_clean[product_lag] == 0].copy()
    test_universe = test_clean[test_clean[product_lag] == 0].copy()
    val_universe = val_clean[val_clean[product_lag] == 0].copy()

    if len(train_universe) == 0 or len(test_universe) == 0:
        print(f"  Not enough customers for target {prod}. Skipping.")
        continue

    features_in_use = [f for f in lag_cols if f != product_lag]

    y_train = train_universe[target_col]
    y_test = test_universe[target_col]
    y_val = val_universe[target_col]

    if y_train.sum() == 0 or y_test.sum() == 0:
        print(f"  No positive class in train/test for {prod}. Skipping.")
        continue

    X_train_full = train_universe[features_in_use]
    X_test_full = test_universe[features_in_use]
    X_val_full = val_universe[features_in_use]

    # --- (b) Feature selection — Random Forest --------------------------
    rf_selector = RandomForestClassifier(
        n_estimators=50, max_depth=5,
        class_weight='balanced', random_state=42,
    )
    rf_selector.fit(X_train_full, y_train)

    df_importance = pd.DataFrame({
        'Variable': features_in_use,
        'Importance': rf_selector.feature_importances_,
    }).sort_values(by='Importance', ascending=False)
    top_7_features = df_importance.head(7)['Variable'].tolist()
    print(f"Top 7 driver variables: {top_7_features}")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_importance.head(7),
        x='Importance', y='Variable', palette='viridis',
    )
    plt.title(f'Purchase Drivers (top features): {prod}')
    plt.xlabel('Relative Importance')
    plt.ylabel('Variables (Lags)')
    plt.tight_layout()
    plt.show()
    plt.close()

    # --- (c) Decision Tree training -------------------------------------
    X_train_opt = X_train_full[top_7_features]
    X_test_opt = X_test_full[top_7_features]
    X_val_opt = X_val_full[top_7_features]

    tree_model = DecisionTreeClassifier(
        max_depth=5, class_weight='balanced', random_state=42,
    )
    tree_model.fit(X_train_opt, y_train)

    y_pred = tree_model.predict(X_test_opt)
    y_proba = tree_model.predict_proba(X_test_opt)[:, 1]

    # --- (d) Metrics + Confusion Matrix + ROC ---------------------------
    print("\n>>> CLASSIFICATION REPORT <<<")
    print(classification_report(
        y_test, y_pred, target_names=['No Purchase', 'Purchase'],
    ))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Confusion Matrix: {prod}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'AUC = {roc_auc:.2f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_title(f'ROC Curve: {prod}')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # Store the model + metadata
    model_results[prod] = {
        'model': tree_model,
        'features': top_7_features,
        'auc': roc_auc,
    }

    # --- (e) SHAP feature impact analysis -------------------------------
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_test_opt)
    # SHAP returns either a list of arrays (one per class) or a 3D ndarray.
    # We need the values for the positive class (index 1).
    if isinstance(shap_values, list):
        shap_values_positive = shap_values[1]
    else:
        shap_values_positive = shap_values[:, :, 1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_positive, X_test_opt, show=False)
    plt.title("Real Variable Impact on Purchase Probability (SHAP)")
    plt.tight_layout()
    plt.show()

    # --- (f) Top 5% real conversion rate on the validation month --------
    y_val_proba = tree_model.predict_proba(X_val_opt)[:, 1]
    df_simulation = pd.DataFrame({
        'pk_cid_n': val_universe['pk_cid_n'].values,
        'model_score': y_val_proba,
        'real_purchase': y_val.values,
    }).sort_values(by='model_score', ascending=False)

    top_5_pct = 0.05
    top_volume = int(len(df_simulation) * top_5_pct)
    top_customers = df_simulation.head(top_volume)
    purchases_in_top = top_customers['real_purchase'].sum()
    real_conversion_rate = purchases_in_top / top_volume

    print("\n--- VALIDATION CONVERSION SIMULATION ---")
    print(f"Total evaluated customers: {len(df_simulation)}")
    print(f"Customers in top {top_5_pct * 100}%: {top_volume}")
    print(f"Real purchases inside the top: {purchases_in_top}")
    print(f"REAL CONVERSION RATE OF THE SEGMENT: {real_conversion_rate * 100:.2f}%")

    model_results[prod]['real_conversion_rate'] = real_conversion_rate
    model_results[prod]['df_simulation'] = df_simulation


# =============================================================================
# 7. PRODUCT SELECTION FOR THE FINAL CAMPAIGN
# =============================================================================
# After running all four models, we drop:
#   - long_term_deposit_b  → very low real conversion rate.
#   - short_term_deposit_b → not enough positive samples in test/val to
#                             evaluate reliably.
# Final campaign products:
#   - pension_plan (primary, biggest margin contributor).
#   - em_acount    (secondary, value-add cross-sell).
if 'short_term_deposit_b' in target_products:
    target_products.remove('short_term_deposit_b')
if 'long_term_deposit_b' in target_products:
    target_products.remove('long_term_deposit_b')
if 'long_term_deposit_b' in model_results:
    del model_results['long_term_deposit_b']

# Mapping of internal column suffixes to product names for margin lookup
target_products_clean = ['pension_plan', 'em_acount']


# =============================================================================
# 8. JUNE 2019 CAMPAIGN PROJECTION
# =============================================================================
# We score every eligible customer (didn't own the product in May 2019)
# with each model, take the top 5% by predicted probability and translate
# that into expected new acquisitions and expected revenue using:
#   expected_acquisitions = top_volume * real_conversion_rate
#   expected_revenue      = expected_acquisitions * mean_margin_per_unit
projection_cols = ['pk_cid_n'] + features_base
df_may_2019 = df_ml[
    (df_ml['pk_partition_month'] == 5) & (df_ml['pk_partition_year'] == 2019)
][projection_cols].copy()

# Reuse the lag column names directly — May features are the input to a
# June prediction.
for col in features_base:
    if col in df_may_2019.columns:
        df_may_2019[f'{col}_lag1'] = df_may_2019[col]

for prod in target_products_clean:
    df_scoring = df_may_2019[df_may_2019[f'{prod}_b_lag1'] == 0].copy()
    X_scoring = df_scoring[model_results[f'{prod}_b']['features']]

    proba_proj = model_results[f'{prod}_b']['model'].predict_proba(X_scoring)[:, 1]
    pred_proj = model_results[f'{prod}_b']['model'].predict(X_scoring)

    df_projection = pd.DataFrame({
        'pk_cid_n': df_scoring['pk_cid_n'],
        'purchase_probability': proba_proj,
    }).sort_values(by='purchase_probability', ascending=False)

    print(f"\n--- TOP 10 CUSTOMERS TO CONTACT IN JUNE 2019 — {prod} ---")
    print(df_projection['pk_cid_n'].head(10))

    model_results[f'{prod}_b']['df_projection'] = df_projection
    model_results[f'{prod}_b']['prediction'] = pred_proj

    top_5_pct = 0.05
    top_volume = int(len(df_projection) * top_5_pct)
    conv_rate = round(model_results[f'{prod}_b']['real_conversion_rate'], 4)
    projected_volume = round(top_volume * model_results[f'{prod}_b']['real_conversion_rate'])
    unit_margin = round(
        top4_margins[top4_margins['product_desc'] == prod]['mean_margin'].item(), 2
    )
    projected_revenue = projected_volume * unit_margin

    print(f"\n--- REVENUE PROJECTION FOR {prod} (JUNE 2019) ---")
    print(f"Customers in top {top_5_pct * 100}%: {top_volume}")
    print(f"Projected acquisitions (conv. rate {round(conv_rate * 100, 2)}%): "
          f"{projected_volume}")
    print(f"Projected revenue: ${projected_revenue}")


# =============================================================================
# 9. CROSS-REFERENCE WITH CUSTOMER SEGMENTS
# =============================================================================
# Bring back the clusters built by Script 2 and check which business
# segments are over-represented inside the top pension_plan candidates.
# This tells the marketing team where to focus the campaign.
print("\nLoading cluster assignments...")
df_clusters = pd.read_parquet(INPUT_CLUSTERS, engine='pyarrow')

df_top_pension = (
    model_results['pension_plan_b']['df_projection']
    .merge(df_clusters, on='pk_cid_n', how='left')
)

# Take the top 21,052 customers (≈ top 5% of pension_plan eligibles)
top_n = min(21052, len(df_top_pension))
df_top_pension = df_top_pension.head(top_n)
print(df_top_pension.head())
print(df_top_pension.info())

print("\nDistribution of business segments in the top pension_plan candidates:")
if 'cluster_name' in df_top_pension.columns:
    print(df_top_pension['cluster_name'].value_counts())
