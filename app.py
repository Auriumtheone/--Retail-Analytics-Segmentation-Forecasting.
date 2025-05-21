

from flask import Flask, render_template, request, redirect, session, url_for
import csv
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key_here')  # Better to use environment variables

# Configuration
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'

# Password configuration
CORRECT_PASSWORD_HASH = generate_password_hash('password')  # Change this password
  
      
    
class Segmenter:
    def __init__(self):
        try:
            print("Attempting to load models...")  # Debug
            self.model = joblib.load(MODELS_DIR / 'segment_kmeans.joblib')
            self.scaler = joblib.load(MODELS_DIR / 'segment_scaler.joblib')
            print("Models loaded successfully!")  # Debug
        except Exception as e:
            print(f"CRITICAL ERROR in __init__: {str(e)}")  # Debug
            raise

       
    
    def predict(self, customer_id):
        try:
            # Load data (ensure correct path)
            new_data = pd.read_csv('purchases.csv', parse_dates=['InvoiceDate'])
            
            # Filter for customer (exact match)
            customer_data = new_data[new_data['Customer_ID'] == int(customer_id)]  # Convert to int
            if customer_data.empty:
                return None
                
            # Calculate RFM (use existing Revenue)
            current_date = pd.to_datetime(datetime.now())  # Use current time
            rfm = pd.DataFrame({
                'Recency': (current_date - customer_data['InvoiceDate'].max()).days,
                'Frequency': customer_data['Invoice'].nunique(),
                'Monetary': customer_data['Revenue'].sum()  # Sum existing Revenue
            }, index=[0])
            
            # Debug: Print RFM values
            print("RFM Values:", rfm)
            
            # Predict (skip if invalid)
            if rfm['Recency'].iloc[0] < 0 or rfm['Monetary'].iloc[0] <= 0:
                return None
                
            rfm_log = np.log1p(rfm)
            rfm_scaled = self.scaler.transform(rfm_log)
            return self.model.predict(rfm_scaled)[0]
            
        except Exception as e:
            print("Error in prediction:", str(e))  # Debug
            return None
    
    


# Initialize segmenter
segmenter = Segmenter()

# Helper functions
def init_csv():
    """Initialize purchases CSV with required columns"""
    if not os.path.exists('purchases.csv'):
        with open('purchases.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Invoice', 'StockCode', 'Description', 'Quantity',
                'InvoiceDate', 'Price', 'Customer_ID', 'Country', 'Revenue'
            ])

def get_purchases():
    """Read existing purchases from CSV"""
    purchases = []
    if os.path.exists('purchases.csv'):
        with open('purchases.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            purchases = list(reader)
    return purchases



@app.route('/test_models')
def test_models():
    try:
        # Test loading models directly
        test_model = joblib.load(MODELS_DIR / 'segment_kmeans.joblib')
        test_scaler = joblib.load(MODELS_DIR / 'segment_scaler.joblib')
        return "Models loaded successfully!"
    except Exception as e:
        return f"Model loading failed: {str(e)}"



# Authentication routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        password = request.form.get('password')
        if password and check_password_hash(CORRECT_PASSWORD_HASH, password):
            session['logged_in'] = True
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid password')
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form['password']
        if check_password_hash(CORRECT_PASSWORD_HASH, password):
            session['logged_in'] = True
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Application routes
@app.route('/index')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/add_purchase', methods=['POST'])
def add_purchase():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        # Get and validate form data
        quantity = int(request.form['quantity'])
        price = float(request.form['price'])
        revenue = quantity * price
        
        # Write to CSV
        with open('purchases.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                request.form['invoice'],
                request.form['stock_code'],
                request.form['description'],
                quantity,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                price,
                request.form['customer_id'],
                request.form['country'],
                revenue
            ])
        return redirect(url_for('index'))
    except Exception as e:
        return f"Error: {str(e)}", 400

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    purchases = get_purchases()
    return render_template('dashboard.html', purchases=purchases)



@app.route('/segment', methods=['POST'])
def segment_customer():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    customer_id = request.form.get('customer_id')
    purchases = get_purchases()
    
    if not customer_id:
        return render_template('dashboard.html',
                           purchases=purchases,
                           error="Please enter Customer ID")
    
    try:
        # Convert to int and verify the customer exists first
        customer_id = int(customer_id)
        
        # Debug: Verify customer exists in purchases
        customer_exists = any(int(row['Customer_ID']) == customer_id for row in purchases)
        if not customer_exists:
            return render_template('dashboard.html',
                               purchases=purchases,
                               error=f"Customer {customer_id} not found in purchases.csv")
        
        # Now try segmentation
        segment = segmenter.predict(customer_id)
        if segment is None:
            return render_template('dashboard.html',
                               purchases=purchases,
                               error="Segmentation failed for valid customer")
        
        segments = {
            0: "Low-Value",
            1: "Mid-Value",
            2: "High-Value",
            3: "Champion"
        }
        
        return render_template('dashboard.html',
                            purchases=purchases,
                            segment_result={
                                'id': customer_id,
                                'segment': segments.get(segment, 'Unknown'),
                                'segment_code': segment
                            })
    except ValueError:
        return render_template('dashboard.html',
                           purchases=purchases,
                           error="Invalid Customer ID (must be a number)")
    except Exception as e:
        return render_template('dashboard.html',
                           purchases=purchases,
                           error=f"Analysis failed: {str(e)}")



if __name__ == '__main__':
    init_csv()
    app.run(debug=True)