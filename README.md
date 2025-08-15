# Clustering-with-K-means
K-Means is an unsupervised machine learning algorithm that partitions data into K clusters based on similarity. Each data point belongs to the cluster with the nearest mean (centroid). In this task, we’ll use the Mall Customers dataset to segment customers into groups for targeted marketing.
Step-by-Step Procedure
1. Import Libraries
Load essential libraries such as Pandas, NumPy, Matplotlib, and scikit-learn’s KMeans and metrics.
2. Load Dataset
Import the Mall Customer dataset from a CSV file.
3. Select Features
Choose relevant numerical features (e.g., Annual Income (k$), Spending Score (1-100)).
4. Optional PCA for Visualization
If your dataset has more than 2 dimensions, use PCA to reduce it to 2D for plotting.
5. Elbow Method
Plot Sum of Squared Errors (SSE) for different K values to determine the optimal number of clusters.
6. Fit K-Means
Apply K-Means with the chosen K and get cluster labels.
7. Visualize Clusters
Create a scatter plot with different colors for each cluster and mark centroids.
8. Evaluate with Silhouette Score
Measure how well-separated the clusters are.
