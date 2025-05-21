import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load and preprocess retail data"""
    df = pd.read_csv(filepath, parse_dates=['InvoiceDate'])
    current_date = df['InvoiceDate'].max() + pd.DateOffset(days=1)
    
    # Calculate RFM
    rfm = df.groupby('Customer_ID').agg({
        'InvoiceDate': lambda x: (current_date - x.max()).days,
        'Invoice': 'nunique',
        'Revenue': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'Revenue': 'Monetary'
    })
    
    return rfm

def train_model(rfm_data, n_clusters=4):
    """Train and save segmentation model"""
    # Log transformation
    rfm_log = np.log1p(rfm_data[['Recency', 'Frequency', 'Monetary']])
    
    # Standardization
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(rfm_scaled)
    
    # Save artifacts
    joblib.dump(kmeans, 'models/segment_kmeans.joblib')
    joblib.dump(scaler, 'models/segment_scaler.joblib')
    
    return kmeans, scaler



if __name__ == "__main__":
    # Append new data to historical
    new_data = pd.read_csv('data/new_transactions.csv')
    historical = pd.read_csv('data/processed/cleaned_online_retail.csv')
    updated_data = pd.concat([historical, new_data])
    updated_data.to_csv('data/processed/cleaned_online_retail.csv', index=False)
    
    # Now train
    rfm_data = load_data('data/processed/cleaned_online_retail.csv')



    plt.figure(figsize=(8,6))
    sns.heatmap(rfm_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("RFM kintamųjų koreliacijų šiluminis žemėlapis")
    plt.show()





    model, scaler = train_model(rfm_data)

    print("Model trained and saved successfully!")
