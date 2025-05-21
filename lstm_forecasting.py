import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# Load Data
file_path = "data/processed/cleaned_online_retail.csv"
df = pd.read_csv(file_path)

# Convert InvoiceDate to datetime & sort
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.sort_values(by='InvoiceDate', inplace=True)

# Group by month and aggregate revenue
df.set_index('InvoiceDate', inplace=True)
df_monthly = df.resample('M').sum()  # Monthly revenue
df_monthly['Revenue'].fillna(df_monthly['Revenue'].median(), inplace=True)


# 🔹 Visualize Revenue Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_monthly['Revenue'], bins=20, kde=True, color="blue")
plt.title("Revenue Distribution")
plt.xlabel("Total Revenue")
plt.ylabel("Frequency")
plt.show()

# 🔹 Bar Chart of Top-Selling Countries
top_countries = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_countries.index, y=top_countries.values, palette="Blues_r")
plt.xticks(rotation=45)
plt.title("Top 10 Countries by Revenue")
plt.ylabel("Total Revenue")
plt.show()

# Normalize data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_monthly[['Revenue']])
print("Scaled data shape:", df_scaled.shape)


# Debugging: Check if data is properly scaled
print("Data shape before sequencing:", df_scaled.shape)
print("First few values:", df_scaled[:5])

print("Total rows in dataset:", len(df_monthly))


# Create sequences for LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 6  # Use previous 12 months to predict the next one
X, y = create_sequences(df_scaled, seq_length)


X, y = create_sequences(df_scaled, seq_length)
print("Generated sequences X shape:", X.shape)
print("Generated sequences y shape:", y.shape)

# Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 🔹 Define LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Debugging: Check training data shape before starting training

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save trained model
model.save("lstm_sales_forecast.h5")

# 🔹 Generate Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Compare Actual vs Predicted Revenue
plt.figure(figsize=(10, 5))
sns.lineplot(x=df_monthly.index[-len(predictions):], y=df_monthly['Revenue'].values[-len(predictions):], label="Actual Revenue", marker="o")
sns.lineplot(x=df_monthly.index[-len(predictions):], y=predictions.flatten(), label="Predicted Revenue", marker="o", linestyle="dashed")
plt.title("LSTM Forecast vs Actual Revenue")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.grid()
plt.show()