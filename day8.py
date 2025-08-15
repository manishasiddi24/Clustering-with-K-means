import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Step 1: Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Step 2: Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Elbow Method
sse = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    sse.append(km.inertia_)

plt.plot(K_range, sse, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 4: Fit K-Means with optimal K (example: K=5)
k_opt = 5
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Step 5: Evaluate
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.2f}")

# Step 6: Visualize Clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='rainbow', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color='black', marker='X', s=200, label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()