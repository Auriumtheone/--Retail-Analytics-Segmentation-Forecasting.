



import pandas as pd
import numpy as np
import joblib
from datetime import datetime

class CustomerSegmenter:
    def __init__(self):
        self.model = joblib.load('models/segment_kmeans.joblib')
        self.scaler = joblib.load('models/segment_scaler.joblib')
    
    def calculate_rfm(self, transaction_data, customer_id):
        """Calculate RFM metrics for a specific customer using your exact format"""
        # Convert to datetime if not already
        transaction_data['InvoiceDate'] = pd.to_datetime(transaction_data['InvoiceDate'])
        
        current_date = transaction_data['InvoiceDate'].max() + pd.DateOffset(days=1)
        
        # Filter for customer and calculate metrics
        customer_data = transaction_data[transaction_data['Customer_ID'] == customer_id]
        
        if customer_data.empty:
            return None
            
        rfm = pd.DataFrame({
            'Recency': (current_date - customer_data['InvoiceDate'].max()).days,
            'Frequency': customer_data['Invoice'].nunique(),
            'Monetary': customer_data['Revenue'].sum()
        }, index=[0])
        
        return rfm
    
    def predict_segment(self, rfm_data):
        """Predict customer segment"""
        # Transformations
        rfm_log = np.log1p(rfm_data)
        rfm_scaled = self.scaler.transform(rfm_log)
        
        # Prediction
        segment = self.model.predict(rfm_scaled)[0]
        return segment

# Example usage with YOUR data format:
if __name__ == "__main__":
    # Load new transaction data (with your exact format)
    try:
        new_data = pd.read_csv('data/new_transactions.csv', 
                             parse_dates=['InvoiceDate'])
        print("Data loaded successfully. Sample:")
        print(new_data.head())
        
        # Get first customer ID in data for testing
        test_customer = new_data['Customer_ID'].iloc[0]
        print(f"\nTesting with customer ID: {test_customer}")
        
        # Initialize and predict
        segmenter = CustomerSegmenter()
        customer_rfm = segmenter.calculate_rfm(new_data, test_customer)
        
        if customer_rfm is not None:
            print("\nCalculated RFM values:")
            print(customer_rfm)
            
            segment = segmenter.predict_segment(customer_rfm)
            print(f"\nPredicted segment: {segment}")
            
            # Segment interpretation guide
            segments = {
                0: "Low-Value",
                1: "Mid-Value",
                2: "High-Value",
                3: "Champions"
            }
            print(f"Segment meaning: {segments.get(segment, 'Unknown')}")
        else:
            print("Error: Customer not found in data")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please verify:")
        print("- File exists at 'data/new_transactions.csv'")
        print("- Columns match: Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer_ID,Country,Revenue")