# Import libraries (add to your existing imports)




import pandas as pd  
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Load your cleaned data
df_cleaned = pd.read_csv("data/processed/cleaned_online_retail.csv", 
                         parse_dates=['InvoiceDate'])

# --- RFM CALCULATION (Combined Approach) ---
# 1. Define snapshot date (your version)
current_date = df_cleaned['InvoiceDate'].max() + pd.DateOffset(days=1)

# 2. Calculate RFM (my optimized grouping + your column names)
rfm = df_cleaned.groupby('Customer_ID').agg({
    'InvoiceDate': lambda x: (current_date - x.max()).days,  # Recency
    'Invoice': 'nunique',                                   # Frequency
    'Revenue': 'sum'                                        # Monetary
}).reset_index()

rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']  # Your naming

# --- CLUSTERING (Your Code Enhanced) ---
# 1. Log-transform (your approach)
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])

# 2. K-means with elbow method for optimal clusters
wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_log)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()





# 3. Apply K-means with selected K (e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_log)


# Skaičiuojame Silhouette Score
score = silhouette_score(rfm_log, rfm['Segment'])
print(f"Silhouette Score: {score:.2f}")

# Individualūs Silhouette koeficientai
silhouette_vals = silhouette_samples(rfm_log, rfm['Segment'])

# Vizualizacija Jupyter Notebook'e
plt.figure(figsize=(10, 5))
y_lower = 10
for i in range(4):  # Jei naudojai K=4
    cluster_silhouette_vals = silhouette_vals[rfm['Segment'] == i]
    cluster_silhouette_vals.sort()
    y_upper = y_lower + len(cluster_silhouette_vals)
    
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals)
    plt.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i))
    y_lower = y_upper + 10

# Raudona linija – bendras Silhouette Score
plt.axvline(score, color="red", linestyle="--", label="Overall Silhouette Score")
plt.title("Silhouette plot for K-Means clustering")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")
plt.legend()
plt.show()



# --- VISUALIZATIONS (New Additions) ---
# 1. 3D RFM Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'], 
                     c=rfm['Segment'], cmap='viridis', s=50)
ax.set_xlabel("Recency (days)")
ax.set_ylabel("Frequency")
ax.set_zlabel("Monetary ($)")
plt.title("3D RFM Segmentation")
plt.colorbar(scatter)
plt.show()

# 2. Pairplot
sns.pairplot(rfm, vars=['Recency', 'Frequency', 'Monetary'], 
             hue='Segment', palette='viridis', corner=True)
plt.suptitle("RFM Pairplot by Segment", y=1.02)
plt.show()

# 3. Segment Summary (your code enhanced)
segment_summary = rfm.groupby('Segment').agg({
    'Recency': ['mean', 'median'],
    'Frequency': ['mean', 'median'],
    'Monetary': ['mean', 'median', 'sum'],
    'Customer_ID': 'count'
}).round(2)






print("\n Segment Summary:")
print(segment_summary.to_string())  # .to_string() ensures clean formatting

# Save results
rfm.to_csv('data/processed/rfm_segments.csv', index=False)