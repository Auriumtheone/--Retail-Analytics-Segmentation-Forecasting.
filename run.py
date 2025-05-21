


import pandas as pd
from sqlalchemy.orm import sessionmaker
from database import engine, Transaction, Customer, Product  # Ensure correct imports

# Load raw Excel data
file_path = "data/raw/online_retail_II.xlsx"
df = pd.read_excel(file_path)

#  Step 1: Remove columns with all NaN values  
df.dropna(how="all", axis=1, inplace=True)  

#  Step 2: Clean missing values safely  
df_cleaned = df.dropna(subset=["Customer ID", "Description"]).copy()

#  Step 3: Rename "Customer ID" for consistency  
df_cleaned.rename(columns={"Customer ID": "Customer_ID"}, inplace=True)

#  Step 4: Convert Customer_ID to numeric and handle NaN  
df_cleaned["Customer_ID"] = pd.to_numeric(df_cleaned["Customer_ID"], downcast="integer", errors="coerce")

#  Step 5: Remove extreme outliers (Quantity & Price must be positive)  
df_cleaned = df_cleaned[(df_cleaned["Quantity"] > 0) & (df_cleaned["Price"] > 0)]

#  Step 6: Add Revenue column  
df_cleaned["Revenue"] = df_cleaned["Quantity"] * df_cleaned["Price"]

#  Step 7: Save cleaned data as CSV  
cleaned_file_path = "C:/Users/Neurogen/Documents/retail_analytics/data/processed/cleaned_online_retail.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)

#  Step 8: Verify remaining missing values  
print("\n Remaining missing values:\n", df_cleaned.isnull().sum())

#  Step 9: Load processed data from CSV  
df_cleaned = pd.read_csv(cleaned_file_path, dtype={"Invoice": str}, low_memory=False)

#  Step 10: Map to database tables  
transactions_list = df_cleaned.to_dict(orient="records")
customers_list = df_cleaned[["Customer_ID"]].drop_duplicates().to_dict(orient="records")
products_list = df_cleaned[["StockCode", "Description", "Price"]].drop_duplicates().to_dict(orient="records")
invoices_list = df_cleaned[["Invoice", "InvoiceDate", "Customer_ID"]].drop_duplicates().to_dict(orient="records")

#  Summary Output  
print(f"\n Total transactions prepared: {len(transactions_list)}")
print(f" Total customers prepared: {len(customers_list)}")
print(f" Total products prepared: {len(products_list)}")




# Įkelkite apdorotus duomenis iš CSV
file_path = "C:/Users/Neurogen/Documents/retail_analytics/data/processed/cleaned_online_retail.csv"
df = pd.read_csv(file_path)

# Išveskite visus stulpelių pavadinimus
print(df.columns)




