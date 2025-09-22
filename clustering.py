import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestCentroid

# ----------------------------------------
# Step 1: Load and (optionally) sample data
# ----------------------------------------
df = pd.read_csv("E:/thesis/user_combined_summary.csv")

MAX_ROWS = 20000
if len(df) > MAX_ROWS:
    print(f" Dataset has {len(df)} rows. Sampling {MAX_ROWS} rows for clustering...")
    df = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
else:
    print(f" Dataset has {len(df)} rows. No sampling needed.")

# ----------------------------------------
# Step 2: Feature Engineering
# ----------------------------------------
drop_cols = ['user_id']
df_features = df.drop(columns=drop_cols)
df_features = df_features.fillna(df_features.mean(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ----------------------------------------
# Step 3: Clustering (KMeans vs Agglomerative)
# ----------------------------------------
n_clusters = 4

# KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
sil_kmeans = silhouette_score(X_scaled, labels_kmeans)

# Agglomerative
agg = AgglomerativeClustering(n_clusters=n_clusters)
labels_agg = agg.fit_predict(X_scaled)
sil_agg = silhouette_score(X_scaled, labels_agg)
print(f"Silhouette Score - Kmeans: {sil_kmeans:.3f}, Agglomerative: {sil_agg:.3f}")
# ----------------------------------------
# Step 4: Choose Best Model
# ----------------------------------------
if sil_kmeans >= sil_agg:
    best_model = "KMeans"
    final_labels = labels_kmeans
    clusterer = kmeans
else:
    best_model = "Agglomerative"
    final_labels = labels_agg
    clusterer = NearestCentroid()
    clusterer.fit(X_scaled, labels_agg)

print(f"\n Best model based on silhouette score: {best_model}")

# ----------------------------------------
# Step 5: Visualization
# ----------------------------------------
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=final_labels, palette='tab10')
plt.title(f"{best_model} Clustering (Silhouette={max(sil_kmeans, sil_agg):.3f})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# ----------------------------------------
# Step 6: Cluster Profiling
# ----------------------------------------
df_clustered = df.copy()
df_clustered['cluster'] = final_labels
profiles = df_clustered.groupby('cluster').mean(numeric_only=True)

print("\n Cluster Profiles (Mean of Each Feature):")
print(profiles)

# ----------------------------------------
# Step 7: Predict Cluster for New Data
# ----------------------------------------

# 👇 Replace this with real/new user data row
new_data = pd.DataFrame([{
    'submission_count': 5,
    'avg_cpu_time': 62,
    'avg_memory': 12000,
    'avg_code_size': 850,
    'avg_total_lines': 30,
    'avg_line_spacing': 1.5,
    'avg_total_comments': 3,
    'avg_if_else_count': 1.0,
    'avg_total_variables': 6,
    'avg_loop_count': 1,
    'avg_capitalized_variable_names': 0.2,
    'avg_percentage_for_loops': 0.5,
    'avg_submission_count_per_problem': 1,
    'accepted_count': 1,
    'runtime_error_count': 0,
    'time_limit_exceeded_count': 0,
    'wrong_answer_count': 0
}])

# Match column order and scale
new_data_scaled = scaler.transform(new_data[df_features.columns])

# Predict cluster
if best_model == "KMeans":
    predicted_cluster = clusterer.predict(new_data_scaled)[0]
else:
    predicted_cluster = clusterer.predict(new_data_scaled)[0]

print(f"\n New developer assigned to Cluster #{predicted_cluster}")
print("\n Behavioral Profile of Assigned Cluster:")
print(profiles.loc[predicted_cluster])
