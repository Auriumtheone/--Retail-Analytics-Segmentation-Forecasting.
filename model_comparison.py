import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = "data/processed/cleaned_online_retail.csv"
df = pd.read_csv(file_path)

# Aggregate sales by product
product_sales = df.groupby('StockCode')['Quantity'].sum().reset_index()

# Identify top 10 most popular products
top_products = product_sales.sort_values(by="Quantity", ascending=False).head(10)

# Identify 10 least popular products
least_products = product_sales.sort_values(by="Quantity", ascending=True).head(10)

# 🔹 Plot bar charts
plt.figure(figsize=(12, 5))
# sns.barplot(x=top_products['StockCode'], y=top_products['Quantity'], palette="Blues")

sns.barplot(x=top_products['StockCode'], y=top_products['Quantity'], hue=top_products['StockCode'], palette="Blues", legend=False)

plt.xticks(rotation=45)
plt.title("Top 10 Most Popular Products (Quantity Sold)")
plt.show()

plt.figure(figsize=(12, 5))
# sns.barplot(x=least_products['StockCode'], y=least_products['Quantity'], palette="Reds")
sns.barplot(x=least_products['StockCode'], y=least_products['Quantity'], hue=least_products['StockCode'], palette="Reds", legend=False)

plt.xticks(rotation=45)
plt.title("10 Least Popular Products (Quantity Sold)")
plt.show()


# Aggregate total quantity sold per product
product_sales = df.groupby('StockCode')['Quantity'].sum().reset_index()

# Get top 50 best-selling products
top_50_products = product_sales.sort_values(by="Quantity", ascending=False).head(50)

# 🔹 Bar Chart of Top 50 Sold Products
plt.figure(figsize=(15, 6))
# sns.barplot(y=top_50_products['StockCode'], x=top_50_products['Quantity'], palette="Blues_r")

sns.barplot(y=top_50_products['StockCode'], x=top_50_products['Quantity'], hue=top_50_products['StockCode'], palette="Blues_r", legend=False)

plt.xlabel("Quantity Sold")
plt.ylabel("Product Code (StockCode)")
plt.title("Top 50 Best-Selling Products")
plt.show()

# 🔹 Pie Chart of Sales Share Among Top 50 Products
plt.figure(figsize=(8, 8))
plt.pie(top_50_products['Quantity'], labels=top_50_products['StockCode'], autopct='%1.1f%%', colors=sns.color_palette("coolwarm", len(top_50_products)))
plt.title("Sales Distribution for Top 50 Best-Selling Products")
plt.show()

