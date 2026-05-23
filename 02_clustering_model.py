"""
TFM — Banking Customer Analytics
=================================
Script 2 of 3 — Customer Segmentation (Clustering)

Loads the prepared dataset built in Script 1, filters the most recent
snapshot (May 2019) of active customers and segments them into business-
meaningful clusters using MiniBatchKMeans. Compares two feature sets and
benchmarks against the traditional KMeans algorithm before assigning
business names to each segment.

Why MiniBatchKMeans?
    The customer base is large (millions of rows over the full historical
    window). MiniBatchKMeans uses small random batches at every iteration,
    which makes it dramatically faster than the standard KMeans without a
    significant loss in quality.

Pipeline:
    1. Load the prepared dataset (parquet, gzip) and filter the snapshot
       (May 2019, active customers only).
    2. Select an initial feature set (demographics + economics + products +
       commercial segment) and drop columns that turn out to be constant.
    3. Standardize features (StandardScaler).
    4. Run the elbow method (inertia vs. K) to pick a reasonable K.
    5. Compare K=5, K=7, K=8 candidate solutions.
    6. Profile clusters with feature-wise means and assign business names.
    7. Re-run the same pipeline with an extended feature set to verify
       segment stability.
    8. Cross-check with traditional KMeans (same K) and produce final
       business names + a persistence file consumed by the propensity model.

Inputs:
    outputs/df_final_preparado.parquet

Output:
    outputs/df_cluster.parquet — per-client cluster assignment for May 2019
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_PATH = "outputs/df_final_preparado.parquet"
OUTPUT_PATH = "outputs/df_cluster.parquet"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# =============================================================================
# 1. LOAD PREPARED DATASET
# =============================================================================
print("Loading prepared dataset...")
df = pd.read_parquet(INPUT_PATH, engine='pyarrow')
print("Dataset shape:", df.shape)
print(df.head())


# =============================================================================
# 2. SAMPLE SELECTION (most recent snapshot, active customers)
# =============================================================================
# We work with the May 2019 snapshot and only active customers. This makes
# the resulting segments commercially actionable — the alternative (all
# months, all customers) would mostly produce a trivial split between
# active and inactive customers.
df_cluster = df[
    (df['pk_partition_year'] == 2019) &
    (df['pk_partition_month'] == 5)
].copy()
df_cluster = df_cluster[df_cluster['active_customer_b'] == 1].copy()
print("Shape after filtering May 2019 + active customers:", df_cluster.shape)


# =============================================================================
# 3. INITIAL FEATURE SELECTION
# =============================================================================
# Demographic + economic + product + commercial-segment variables.
# 'active_customer_b' is excluded on purpose: we already filtered on it, so
# it would be a constant feature contributing nothing.
features = [
    'age_n', 'salary_n', 'net_margin_n',
    'credit_card_b', 'payroll_b', 'pension_plan_b', 'debit_card_b',
    'em_acount_b', 'payroll_account_b', 'emc_account_b',
    'funds_b', 'securities_b', 'long_term_deposit_b',
    'segment_cat_02 - PARTICULARES', 'segment_cat_03 - UNIVERSITARIO',
    'segment_cat_Other',
]
# Keep only features that actually exist in the dataset (defensive)
features = [f for f in features if f in df_cluster.columns]

X = df_cluster[features].copy()

# Drop constant columns: with only May 2019 active customers, some variables
# can be 100% identical → no information for the algorithm.
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
X = X.drop(columns=constant_cols)
features = X.columns.tolist()
print("Constant variables removed:", constant_cols)
print("Final variables used:", features)


# =============================================================================
# 4. FEATURE SCALING
# =============================================================================
# k-means is distance-based, so features must share a common scale. Salary
# and margin would otherwise dominate every cluster boundary.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =============================================================================
# 5. ELBOW METHOD — choosing K
# =============================================================================
# Plot the inertia as K grows. The "elbow" — the point where adding another
# cluster stops producing a meaningful drop in inertia — is a reasonable
# trade-off between detail and interpretability.
inertias = []
K = range(2, 11)
for k in K:
    model = MiniBatchKMeans(
        n_clusters=k, random_state=42, batch_size=10000, n_init=10,
    )
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertias, marker='o')
plt.title('Elbow Method — Initial features')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.show()


# =============================================================================
# 6. CANDIDATE SOLUTIONS (K = 5, 7, 8)
# =============================================================================
# The course guide suggested working with a larger number of segments, so
# we test three candidate Ks before committing to one. Distribution and
# interpretability decide the winner.
for k in [5, 7, 8]:
    model = MiniBatchKMeans(
        n_clusters=k, random_state=42, batch_size=10000, n_init=10,
    )
    df_cluster[f'cluster_{k}'] = model.fit_predict(X_scaled)
    print(f"\nDistribution for K={k}:")
    print(df_cluster[f'cluster_{k}'].value_counts(normalize=True).sort_index().round(3))


# --- Cluster distribution (K=5) ---------------------------------------------
cluster_counts = df_cluster['cluster_5'].value_counts().sort_index()
print("\nK=5 cluster sizes:")
print(cluster_counts)

cluster_counts.plot(kind='bar')
plt.title("Customer distribution by cluster (K=5)")
plt.xlabel("Cluster")
plt.ylabel("Number of customers")
plt.show()


# =============================================================================
# 7. CLUSTER PROFILING (K=5, initial features)
# =============================================================================
profile_clusters = df_cluster.groupby('cluster_5')[features].mean().round(2)
print("\nK=5 cluster profiles (initial features):")
print(profile_clusters)


# -----------------------------------------------------------------------------
# Initial-features interpretation (K=5)
# -----------------------------------------------------------------------------
# The main separation between segments is NOT salary (similar across all
# clusters) but rather age, activity level, generated margin and product
# uptake.
#
# Two low-engagement groups stand out:
#   - Mature customers with very low activity (cluster 0).
#   - Young customers with a very basic relationship (cluster 1).
# Both show low profitability and minimal product penetration.
#
# Cluster 2 concentrates the highest-value customers — top net margin,
# strong presence of payroll and pension plans → high commercial bonding.
# Cluster 3 = active transactional customers (debit + checking account).
# Cluster 4 = active but still very basic, reduced margin, low diversification.
#
# Business value: differentiated strategies per segment (activation for the
# low-engagement clusters, cross-selling for the active basics, retention
# for the high-value customers).
# -----------------------------------------------------------------------------

cluster_names_v1 = {
    0: 'Low-engagement mature customers',
    1: 'Young basic customers',
    2: 'High-value, highly engaged customers',
    3: 'Active transactional customers',
    4: 'Active basic customers with checking account',
}
df_cluster['cluster_name_v1'] = df_cluster['cluster_5'].map(cluster_names_v1)
print("\nClient distribution by business segment (initial features):")
print(df_cluster['cluster_name_v1'].value_counts())

df_cluster['cluster_name_v1'].value_counts().plot(kind='bar')
plt.title("Customer distribution by business segment")
plt.xlabel("Segment")
plt.ylabel("Number of customers")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# =============================================================================
# 8. SECOND EXPERIMENT — Extended feature set
# =============================================================================
# We re-run the pipeline with a broader product mix (adding mortgage and
# removing the segment dummies) to test segment stability. If the clusters
# change drastically, our segmentation is not robust; if they stay similar,
# we have a sign of stability.
features_v2 = [
    'age_n', 'salary_n', 'net_margin_n',
    'credit_card_b', 'payroll_b', 'pension_plan_b', 'debit_card_b',
    'em_acount_b', 'mortgage_b', 'funds_b', 'securities_b',
    'long_term_deposit_b', 'payroll_account_b', 'emc_account_b',
]
features_v2 = [col for col in features_v2 if col in df_cluster.columns]
X_v2 = df_cluster[features_v2].copy()

scaler_v2 = StandardScaler()
X_v2_scaled = scaler_v2.fit_transform(X_v2)

# Elbow with extended features
inertias_v2 = []
for k in K:
    model = MiniBatchKMeans(
        n_clusters=k, random_state=42, batch_size=10000, n_init=10,
    )
    model.fit(X_v2_scaled)
    inertias_v2.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertias_v2, marker='o')
plt.title('Elbow Method — Extended features')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.show()


# --- Final model with extended features --------------------------------------
k_final = 5
kmeans_final = MiniBatchKMeans(
    n_clusters=k_final, random_state=42, batch_size=10000, n_init=10,
)
df_cluster['cluster'] = kmeans_final.fit_predict(X_v2_scaled)

# Profile of the extended-feature clusters
profile_clusters = df_cluster.groupby('cluster')[features_v2].mean().round(2)
print("\nCluster profiles (extended features):")
print(profile_clusters)


# =============================================================================
# 9. COMPARISON — initial vs. extended feature set
# =============================================================================
# Cross-tab of cluster assignments between the two models. If the new
# clusters mostly contain customers from a single original cluster, the
# segmentation is stable to feature-set changes.
comparison_clusters = pd.crosstab(
    df_cluster['cluster'], df_cluster['cluster_5'], normalize='index',
).round(2)
print("\nComparison: extended features (rows) vs. initial features (cols):")
print(comparison_clusters)


# =============================================================================
# 10. ALGORITHM SANITY CHECK — Traditional KMeans
# =============================================================================
# MiniBatchKMeans is an approximation of KMeans optimized for large datasets.
# We run the standard KMeans on the same scaled data to confirm the
# segmentation is similar — this validates our choice of the faster variant.
kmeans_traditional = KMeans(n_clusters=5, random_state=42, n_init=10)
df_cluster['cluster_kmeans'] = kmeans_traditional.fit_predict(X_v2_scaled)

profile_kmeans = df_cluster.groupby('cluster_kmeans')[features_v2].mean().round(2)
print("\nProfile — traditional KMeans:")
print(profile_kmeans)


# =============================================================================
# 11. FINAL BUSINESS SEGMENT NAMES
# =============================================================================
# Cluster 0 — Account + EMC mature customers
#   Mean age ~44, heavy use of em_acount and emc_account, low margin.
#   Some maturity but low additional bonding.
#
# Cluster 1 — Young customers with basic checking account
#   Low age, very low margin, basic checking account but barely any other
#   product. Long-term growth potential.
#
# Cluster 2 — High-net-worth investors
#   High salary, high age, 100% with funds and a strong presence of
#   long-term deposits and securities. High profitability, investor profile.
#
# Cluster 3 — High-bonding payroll customers
#   Highest margin, almost all with payroll, pension plan, payroll account
#   and frequent debit usage. Strong commercial bonding.
#
# Cluster 4 — High-salary low-bonding customers
#   Very high salary but very low margin and few products contracted.
#   Major untapped commercial opportunity.

cluster_names = {
    0: 'Mature account + EMC customers',
    1: 'Young customers with basic checking account',
    2: 'High-net-worth investors',
    3: 'High-bonding payroll customers',
    4: 'High-salary low-bonding customers',
}
df_cluster['cluster_name'] = df_cluster['cluster'].map(cluster_names)
print("\nFinal cluster name mapping:")
print(df_cluster[['cluster', 'cluster_name']].drop_duplicates().sort_values('cluster'))

print("\nFinal segment distribution (% of customers):")
print((df_cluster['cluster_name']
       .value_counts(normalize=True)
       .sort_values(ascending=False) * 100).round(2))


# =============================================================================
# 12. PERSIST CLUSTER ASSIGNMENTS
# =============================================================================
# Save only the columns the propensity model needs to cross-reference.
keep_cols = ['pk_cid_n', 'cluster', 'cluster_name']
keep_cols = [c for c in keep_cols if c in df_cluster.columns]
df_cluster[keep_cols].to_parquet(OUTPUT_PATH, compression='gzip', index=False)
print(f"\nCluster assignments saved to {OUTPUT_PATH}")
