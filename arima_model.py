import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings
import statsmodels.api as sm


warnings.filterwarnings("ignore")

# Load Data
file_path = "data/processed/cleaned_online_retail.csv"
df = pd.read_csv(file_path)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Aggregate revenue over time
df.set_index('InvoiceDate', inplace=True)
df = df.resample('M').sum()  # Monthly revenue

# Plot initial revenue trend
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x=df.index, y='Revenue', marker='o', label="Revenue Trend")
plt.title("Monthly Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Total Revenue")
plt.grid()
plt.show()

# Train ARIMA Model
p, d, q = 5, 1, 2  # You can experiment with these parameters
model = ARIMA(df['Revenue'], order=(p, d, q))
model_fit = model.fit()

# Make Predictions
df['Forecast'] = model_fit.predict(start=len(df)-12, end=len(df), dynamic=True)

# Plot Predictions vs Actual Revenue
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x=df.index, y='Revenue', marker='o', label="Actual Revenue")
sns.lineplot(data=df, x=df.index, y='Forecast', marker='o', linestyle="dashed", label="Forecasted Revenue")
plt.title("ARIMA Model: Revenue Prediction")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.grid()
plt.show()