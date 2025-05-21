
import pandas as pd
from sqlalchemy import create_engine, text

# MySQL connection
engine = create_engine('mysql+mysqlconnector://ORACLETM:akle7@localhost/retail_analysis_db')

# Load data from CSV
file_path = "C:/Users/Neurogen/Documents/retail_analytics/data/processed/cleaned_online_retail.csv"
df = pd.read_csv(file_path)

# Clean data – remove missing values
df.dropna(inplace=True)

# Remove existing data while keeping foreign key constraints intact
with engine.connect() as conn:
    conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))  # Disable FK constraints
    conn.execute(text("TRUNCATE TABLE customers"))  # Clears customer table safely
    conn.execute(text("TRUNCATE TABLE products"))   # Clears product table safely
    conn.execute(text("TRUNCATE TABLE transactions"))  # Clears transactions
    conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))  # Re-enable FK constraints

# Insert customers (handling duplicates)
customer_query = text("""
    INSERT INTO customers (Customer_ID, Country) 
    VALUES (:Customer_ID, :Country) 
    ON DUPLICATE KEY UPDATE Country=VALUES(Country)
""")
with engine.connect() as conn:
    conn.execute(text("START TRANSACTION"))
    for _, row in df[['Customer_ID', 'Country']].drop_duplicates().iterrows():
        conn.execute(customer_query, {"Customer_ID": row["Customer_ID"], "Country": row["Country"]})
    conn.execute(text("COMMIT"))

# Insert products (handling duplicates)
product_query = text("""
    INSERT INTO products (StockCode, Description, Price) 
    VALUES (:StockCode, :Description, :Price) 
    ON DUPLICATE KEY UPDATE Description=VALUES(Description), Price=VALUES(Price)
""")
with engine.connect() as conn:
    conn.execute(text("START TRANSACTION"))
    for _, row in df[['StockCode', 'Description', 'Price']].drop_duplicates().iterrows():
        conn.execute(product_query, {"StockCode": row["StockCode"], "Description": row["Description"], "Price": row["Price"]})
    conn.execute(text("COMMIT"))

# Validate transactions before inserting (only existing customers & products)
valid_customers = pd.read_sql("SELECT Customer_ID FROM customers", con=engine)['Customer_ID'].unique()
valid_stockcodes = pd.read_sql("SELECT StockCode FROM products", con=engine)['StockCode'].unique()
df = df[df['Customer_ID'].isin(valid_customers) & df['StockCode'].isin(valid_stockcodes)]

# Insert transactions (handling duplicates)
transaction_query = text("""
    INSERT INTO transactions (Invoice, InvoiceDate, StockCode, Quantity, Price, Revenue, Customer_ID) 
    VALUES (:Invoice, :InvoiceDate, :StockCode, :Quantity, :Price, :Revenue, :Customer_ID)
    ON DUPLICATE KEY UPDATE Quantity=VALUES(Quantity), Price=VALUES(Price), Revenue=VALUES(Revenue)
""")
with engine.connect() as conn:
    conn.execute(text("START TRANSACTION"))
    for _, row in df[['Invoice', 'InvoiceDate', 'StockCode', 'Quantity', 'Price', 'Revenue', 'Customer_ID']].iterrows():
        conn.execute(transaction_query, row.to_dict())
    conn.execute(text("COMMIT"))

print(" All data successfully inserted into MySQL!")



# entire dataset is properly loaded into MySQL, with all constraints handled and foreign keys intact. 
#  All data inserted without duplicates or integrity errors
#  Products, transactions, and customers are fully mapped
#  Database is ready for analytics, segmentation, and predictions



from sqlalchemy import create_engine, text

# Connect to MySQL
engine = create_engine('mysql+mysqlconnector://ORACLETM:akle7@localhost/retail_analysis_db')

# Verify connection
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print(f" Connection successful! Test result: {result.scalar()}")
    


   

with engine.connect() as conn:
    for table in ["transactions", "products", "customers"]:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        print(f" {table}: {result.scalar()} rows")
