
# import pandas as pd
# from sqlalchemy.orm import sessionmaker
# from database import engine, Transaction, Customer, Product  # Ensure correct imports

# # Load raw Excel data
# file_path = "data/raw/online_retail_II.xlsx"
# df = pd.read_excel(file_path)

# #  Step 1: Remove columns with all NaN values  
# df.dropna(how="all", axis=1, inplace=True)  

# #  Step 2: Clean missing values safely  
# df_cleaned = df.dropna(subset=["Customer ID", "Description"]).copy()

# #  Step 3: Rename "Customer ID" for consistency  
# df_cleaned.rename(columns={"Customer ID": "Customer_ID"}, inplace=True)

# #  Step 4: Convert Customer_ID to numeric and handle NaN  
# df_cleaned["Customer_ID"] = pd.to_numeric(df_cleaned["Customer_ID"], downcast="integer", errors="coerce")

# #  Step 5: Remove extreme outliers (Quantity & Price must be positive)  
# df_cleaned = df_cleaned[(df_cleaned["Quantity"] > 0) & (df_cleaned["Price"] > 0)]

# #  Step 6: Add Revenue column  
# df_cleaned["Revenue"] = df_cleaned["Quantity"] * df_cleaned["Price"]

# #  Step 7: Save cleaned data as CSV  
# cleaned_file_path = "C:/Users/Neurogen/Documents/retail_analytics/data/processed/cleaned_online_retail.csv"
# df_cleaned.to_csv(cleaned_file_path, index=False)

# #  Step 8: Verify remaining missing values  
# print("\n Remaining missing values:\n", df_cleaned.isnull().sum())

# #  Step 9: Load processed data from CSV  
# df_cleaned = pd.read_csv(cleaned_file_path, dtype={"Invoice": str}, low_memory=False)

# print("\n Data processing completed successfully!")
# print(f" Total processed transactions: {len(df_cleaned)}")
# print(f" Unique customers detected: {df_cleaned['Customer_ID'].nunique()}")
# print(f" Unique products detected: {df_cleaned['StockCode'].nunique()}")
# print(" Cleaned data saved to:", cleaned_file_path)



import pandas as pd

# Load raw Excel data
file_path = "data/raw/online_retail_II.xlsx"
df = pd.read_excel(file_path)

print("🔹 Raw data loaded. Initial shape:", df.shape)

# Step 1: Remove columns with all NaN values  
df.dropna(how="all", axis=1, inplace=True)  
print(" Removed fully empty columns. New shape:", df.shape)

# Step 2: Clean missing values safely  
df_cleaned = df.dropna(subset=["Customer ID", "Description"]).copy()
print(" Removed rows with missing Customer ID/Description. Rows left:", len(df_cleaned))

# Step 3: Rename "Customer ID" for consistency  
df_cleaned.rename(columns={"Customer ID": "Customer_ID"}, inplace=True)

# Step 4: Convert Customer_ID to numeric and handle NaN  
df_cleaned["Customer_ID"] = pd.to_numeric(df_cleaned["Customer_ID"], downcast="integer", errors="coerce")

# Step 5: Remove extreme outliers (Quantity & Price must be positive)  
df_cleaned = df_cleaned[(df_cleaned["Quantity"] > 0) & (df_cleaned["Price"] > 0)]
print(" Removed invalid transactions (Qty/Price ≤ 0). Rows left:", len(df_cleaned))

# Step 6: Add Revenue column  
df_cleaned["Revenue"] = df_cleaned["Quantity"] * df_cleaned["Price"]

#  NEW: Additional Optimizations
# 1. Parse InvoiceDate as datetime
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
print(" Converted InvoiceDate to datetime format")

# 2. Ensure Invoice column is string before filtering
df_cleaned['Invoice'] = df_cleaned['Invoice'].astype(str)  # Critical fix
initial_count = len(df_cleaned)
df_cleaned = df_cleaned[~df_cleaned['Invoice'].str.startswith('C', na=False)]
print(f" Removed {initial_count - len(df_cleaned)} cancellation transactions")

# 3. Memory optimization
initial_mem = df_cleaned.memory_usage(deep=True).sum() / 1024**2  # MB
df_cleaned = df_cleaned.astype({
    'Customer_ID': 'int32',
    'Quantity': 'int16',
    'Price': 'float32'
})
optimized_mem = df_cleaned.memory_usage(deep=True).sum() / 1024**2
print(f" Memory optimized: {initial_mem:.2f}MB → {optimized_mem:.2f}MB (-{(1-optimized_mem/initial_mem)*100:.1f}%)")

# Save cleaned data as CSV  
cleaned_file_path = "data/processed/cleaned_online_retail.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)

# Final Summary
print("\n FINAL PROCESSING RESULTS:")
print(f" Total processed transactions: {len(df_cleaned):,}")
print(f" Unique customers: {df_cleaned['Customer_ID'].nunique():,}")
print(f" Unique products: {df_cleaned['StockCode'].nunique():,}")
print(f" Date range: {df_cleaned['InvoiceDate'].min().date()} to {df_cleaned['InvoiceDate'].max().date()}")
print(f" Cleaned data saved to: {cleaned_file_path}")