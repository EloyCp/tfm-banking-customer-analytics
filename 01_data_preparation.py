"""
TFM — Banking Customer Analytics
=================================
Script 1 of 3 — Data Preparation

Loads the raw banking datasets, performs Data Understanding diagnostics,
imputes missing values with a column-aware strategy, removes deceased
clients and constant-value columns, engineers date and numerical features,
applies One-Hot Encoding to categorical variables, and persists the final
analytical dataset as a parquet file consumed by the clustering and
propensity models.

Pipeline:
    1. Data understanding (shape, types, nulls, duplicates, value counts).
    2. Null handling per variable family ('Other' placeholder, mode, median).
    3. Merge customer + sales aggregated per month for the net margin target.
    4. Remove deceased clients and zero-variance / single-market columns.
    5. Automatic column classification (dates / binary / numerical /
       categorical).
    6. Date decomposition (year + month) with `-1` placeholder for nulls.
    7. Outlier capping on continuous numericals (salary at p95, age 18-90)
       followed by median imputation.
    8. Top-7 + 'Otros' grouping on high-cardinality categoricals before
       One-Hot Encoding (memory-aware, int8 dummies).
    9. Boolean homogenization (text gender → int, products → int8).
    10. Persist the prepared dataset (parquet, gzip).

Inputs:
    data/customer_new.parquet         — merged customer table
    data/sales.csv                    — transactional sales
    data/product_description.csv      — product catalog

Output:
    outputs/df_final_preparado.parquet
"""

# =============================================================================
# IMPORTS
# =============================================================================
import gc
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_CUSTOMER = "data/customer_new.parquet"
INPUT_SALES = "data/sales.csv"
INPUT_PRODUCTS = "data/product_description.csv"
OUTPUT_PATH = "outputs/df_final_preparado.parquet"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("Loading datasets...")
df_customer = pd.read_parquet(INPUT_CUSTOMER, engine='pyarrow')
df_sales = pd.read_csv(INPUT_SALES)
df_product_desc = pd.read_csv(INPUT_PRODUCTS)

datasets = {
    "customer": df_customer,
    "sales": df_sales,
    "product_desc": df_product_desc,
}


# =============================================================================
# 2. DATA UNDERSTANDING
# =============================================================================
def check_data_understanding(df, name):
    """Print a quick diagnostic report on a dataframe: dimensions, duplicates,
    unique values, dtypes/null counts and a small sample."""
    print(f"\n{'=' * 30}\nANALYSIS OF: {name.upper()}\n{'=' * 30}")
    print(f" Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print(f" Duplicate rows: {df.duplicated().sum()}")
    print(f"\n Unique values per column:\n{df.nunique()}")

    info_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Nulls': df.isnull().sum(),
        'Nulls %': (df.isnull().sum() / len(df) * 100).round(2),
    })
    print("\n Column info and nulls:")
    print(info_df)

    if not df.select_dtypes(include=[np.number]).empty:
        print("\n Numerical summary:")
        print(df.describe().T.round(2))

    print("\n First 3 rows:")
    print(df.head(3))


for name, df in datasets.items():
    check_data_understanding(df, name)


# -----------------------------------------------------------------------------
# Findings from the Data Understanding phase
# -----------------------------------------------------------------------------
# 1. Customers
#    - Age has implausible extremes: clients with age 2 and others over 100.
#      The children are likely savings accounts opened by their parents, but
#      they distort every age-based statistic. Filtered out via age capping.
#
# 2. Salary
#    - About 25% of customers have a missing salary.
#    - A few customers report ~28 million €, while the bulk sit around 88k €.
#      That single outlier shifts the mean dramatically, so we cap salaries
#      at the 95th percentile.
#
# 3. entry_channel
#    - Categorical bank codes (web, branch, partner, etc.) with hundreds of
#      distinct values. We will keep the most frequent ones and group the
#      tail into 'Otros'.
#
# 4. Dates & dummy columns
#    - 'Unnamed: 0' columns are leaked CSV indices → dropped.
#    - `pk_partition` is stored as text — we parse it to extract year/month.
# -----------------------------------------------------------------------------


# --- High cardinality / low variance inspections ----------------------------
print("entry_channel value counts:")
print(df_customer['entry_channel'].value_counts())

print("\nsegment value counts:")
print(df_customer['segment'].value_counts(dropna=False))

print("\nproduct families:")
print(df_product_desc['family_product'].unique())


# --- Geographic representativity --------------------------------------------
# 99.96% of records belong to Spain ('ES'), so country_id has no segmentation
# power. We will drop it later.
percentages = df_customer['country_id'].value_counts(normalize=True) * 100
print("\ncountry_id distribution (%):")
print(percentages)

print("\nregion_code value counts:")
print(df_customer['region_code'].value_counts(dropna=False).head(20))


# =============================================================================
# 3. NULL HANDLING
# =============================================================================
# Imputation strategy by variable family:
#   - Categorical with meaningful 'missingness'  → fill with 'Other'
#     (region_code, entry_channel, segment).
#   - Boolean with a dominant mode               → fill with the mode
#     (gender, payroll, pension_plan).
#   - Continuous, robust against outliers        → fill with the median
#     (salary).

for name, df in datasets.items():
    nulls_count = df.isnull().sum()
    if nulls_count.sum() > 0:
        print(f"\nDataset {name} — null counts per column:")
        print(nulls_count[nulls_count > 0])


def universal_cleaner(df):
    """Column-aware null imputation. Only acts on columns that actually
    exist in the dataframe, so the same function can be reused across the
    five raw datasets without breaking on missing columns."""
    to_other = ['region_code', 'entry_channel', 'segment']
    to_mode = ['gender', 'payroll', 'pension_plan']
    to_median = ['salary']

    for col in df.columns:
        if col in to_other:
            df[col] = df[col].fillna('Other')
        elif col in to_mode and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
        elif col in to_median:
            df[col] = df[col].fillna(df[col].median())
    return df


df_customer = universal_cleaner(df_customer)
df_sales = universal_cleaner(df_sales)
df_product_desc = universal_cleaner(df_product_desc)

# Final null check on the critical columns
print("\nNull check after cleaning:")
print(f"- salary:        {df_customer['salary'].isnull().sum()}")
print(f"- entry_channel: {df_customer['entry_channel'].isnull().sum()}")
print(f"- gender:        {df_customer['gender'].isnull().sum()}")
print(f"- segment:       {df_customer['segment'].isnull().sum()}")


# =============================================================================
# 4. MERGE WITH SALES (target: net margin per customer/month)
# =============================================================================
# The single customer table already contains the merged sociodemographic +
# commercial activity + product information per (pk_cid, pk_partition).
# We only need to bring the monthly net margin from sales.

df = df_customer.copy()
del df_customer

# Drop leaked CSV index columns
cols_to_drop = [c for c in df.columns if 'Unnamed' in c]
df = df.drop(columns=cols_to_drop)

# Ensure consistent column naming on sales
sales_drops = [c for c in df_sales.columns if 'Unnamed' in c]
df_sales = df_sales.drop(columns=sales_drops)
if 'cid' in df_sales.columns:
    df_sales = df_sales.rename(columns={'cid': 'pk_cid'})

# Match temporal grain (monthly period)
df['month_key'] = pd.to_datetime(df['pk_partition']).dt.to_period('M')
df_sales['month_key'] = pd.to_datetime(df_sales['month_sale']).dt.to_period('M')

# Aggregate sales: sum of net margin per (client, month)
sales_monthly = (
    df_sales.groupby(['pk_cid', 'month_key'])['net_margin'].sum().reset_index()
)

# Master merge: left-join sales onto the customer table. Customers with no
# purchases in a given month will have NaN, which we replace with 0.
df = pd.merge(df, sales_monthly, on=['pk_cid', 'month_key'], how='left')
df['net_margin'] = df['net_margin'].fillna(0)
df.drop(columns=['month_key'], inplace=True, errors='ignore')
print(f"Merge complete. Final table rows: {df.shape[0]}")


# =============================================================================
# 5. REMOVE DECEASED CLIENTS AND ZERO-VARIANCE COLUMNS
# =============================================================================
# Deceased customers shouldn't be in any future campaign — drop them first
# and then drop the column itself.
if 'deceased' in df.columns:
    print(f"Rows before removing deceased: {len(df)}")
    df = df[df['deceased'] == 'N']
    print(f"Rows after removing deceased:  {len(df)}")

# Drop columns deemed irrelevant for modeling:
#   - em_account_pp : single constant value (zero variance).
#   - deceased      : after removing deceased customers, all rows say 'N'.
#   - country_id    : 99.96% of records are 'ES' (extreme concentration).
#   - Unnamed: 0    : leaked CSV indices.
cols_to_drop = ['em_account_pp', 'deceased', 'country_id', 'Unnamed: 0']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])


# =============================================================================
# 6. AUTOMATIC COLUMN CLASSIFICATION
# =============================================================================
def auto_classify_columns(df):
    """Classify columns into dates / booleans / numerical / categorical so
    each family can be processed with a dedicated strategy."""
    list_dates = [
        col for col in df.columns
        if any(x in col.lower() for x in ['date', 'partition', 'month_sale'])
    ]
    list_binary = [
        col for col in df.columns
        if df[col].nunique() == 2 and col not in list_dates
    ]
    list_numerical = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and col not in list_binary
        and col not in list_dates
    ]
    list_categorical = [
        col for col in df.columns
        if col not in list_dates
        and col not in list_binary
        and col not in list_numerical
    ]
    return list_dates, list_binary, list_numerical, list_categorical


dates, bins, nums, cats = auto_classify_columns(df)
print(f"\nDates:        {dates}")
print(f"Booleans:     {bins}")
print(f"Numerical:    {nums}")
print(f"Categorical:  {cats}")


# =============================================================================
# 7. DATE PROCESSING
# =============================================================================
# Decompose each date column into year + month integer features. Missing
# values (e.g. 'sin_venta' tags) get a `-1` placeholder so the model can
# distinguish "no sale" from a real date without losing the row.
for col in dates:
    temp_dt = pd.to_datetime(df[col], errors='coerce')
    df[f'{col}_year'] = temp_dt.dt.year.fillna(-1).astype('int16')
    df[f'{col}_month'] = temp_dt.dt.month.fillna(-1).astype('int8')
    df.drop(columns=[col], inplace=True)
    del temp_dt

gc.collect()
print("\nDate columns processed:")
print(df.filter(like='_year').columns.tolist())
print(df.filter(like='_month').columns.tolist())


# =============================================================================
# 8. NUMERICAL PROCESSING (outlier capping + median imputation)
# =============================================================================
# Continuous variables are sensitive to extreme values: salary has a few
# multi-million-euro outliers and age has impossible records (2 yrs, 100+).
# We apply capping (salary at p95, age clipped to [18, 90]) before median
# imputation, so the median itself is not biased by the outliers.
for col in nums:
    temp_series = pd.to_numeric(df[col], errors='coerce')

    if 'salary' in col.lower():
        limit_upper = temp_series.quantile(0.95)
        temp_series = temp_series.clip(upper=limit_upper)
    elif 'age' in col.lower():
        # Cap to a realistic adult banking range
        temp_series = temp_series.clip(lower=18, upper=90)

    temp_series = temp_series.round(2)
    median_val = temp_series.median()
    df[f'{col}_n'] = temp_series.fillna(median_val).astype('float32')
    df.drop(columns=[col], inplace=True)
    del temp_series

gc.collect()


# =============================================================================
# 9. CATEGORICAL PROCESSING (top 7 + Otros, then One-Hot Encoding)
# =============================================================================
# High-cardinality categoricals would explode the column count under naive
# OHE. We keep the 7 most frequent values per column and group the rest
# into 'Otros' before encoding. int8 dummies keep memory under control.
for col in cats:
    print(f"Transforming categorical: {col}")
    top_7 = df[col].value_counts().nlargest(7).index.tolist()
    temp_grouped = df[col].where(df[col].isin(top_7), 'Otros')

    dummies = pd.get_dummies(
        temp_grouped, prefix=f"{col}_cat", drop_first=False, dtype='int8'
    )
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[col], inplace=True)

    del temp_grouped, dummies
    gc.collect()


# =============================================================================
# 10. BOOLEAN PROCESSING (gender mapping + int8 conversion)
# =============================================================================
# 'gender' is the only string-typed boolean and needs to be mapped first.
# The rest of the booleans are already 0/1 but stored as float — we cast
# them to int8 (treating nulls as 0 = no contract / no flag).
if 'gender' in df.columns:
    df['gender_b'] = df['gender'].map({'V': 1, 'H': 0}).fillna(0).astype('int8')
    df.drop(columns=['gender'], inplace=True)

for col in bins:
    if col in df.columns:
        df[f'{col}_b'] = (
            pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int8')
        )
        df.drop(columns=[col], inplace=True)

gc.collect()


# =============================================================================
# 11. PERSIST FINAL DATASET
# =============================================================================
print(f"\nFinal dataset shape: {df.shape}")
print("\nFinal dtypes:")
print(df.dtypes.value_counts())

df.to_parquet(OUTPUT_PATH, compression='gzip', index=False)
print(f"\nSaved to {OUTPUT_PATH}")
