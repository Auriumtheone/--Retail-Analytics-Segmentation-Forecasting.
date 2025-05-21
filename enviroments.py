import sys
import os
import platform
import subprocess

# Get Python version
print(f"Python Version: {sys.version}")

# Get active environment (Works for virtual environments)
print(f"Virtual Environment: {sys.prefix}")

# Get OS details
print(f"Operating System: {platform.system()} {platform.release()} ({platform.version()})")
print(f"Architecture: {platform.architecture()[0]}")

# Get installed packages
try:
    installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode("utf-8")
    print("\nInstalled Packages:\n", installed_packages)
except Exception as e:
    print("Error fetching installed packages:", e)

# Get environment variables
print("\nEnvironment Variables:")
for key, value in os.environ.items():
    print(f"{key}: {value}")



###################################################




# import tensorflow as tf
# print("Available GPUs:", tf.config.list_physical_devices('GPU'))



# import tensorflow as tf
# import flask
# import mysql.connector
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from flask import Flask

# #  1. Check TensorFlow Setup
# print("\n TensorFlow Version:", tf.__version__)
# print(" TensorFlow Device List:", tf.config.list_physical_devices('GPU' if tf.config.list_physical_devices('GPU') else 'CPU'))

# # Simple TensorFlow operation
# x = tf.constant([[2.0, 3.0]])
# w = tf.Variable([[4.0], [5.0]])
# y = tf.matmul(x, w)
# print("\n TensorFlow Matrix Multiplication Result:\n", y.numpy())

# #  2. Check MySQL Connection
# try:
#     conn = mysql.connector.connect(
#         host="localhost",
#         user="ORACLETM",      # Change to your MySQL username
#         password="akle7",  # Change to your MySQL password
#         database="test_db"  # Make sure the database exists
#     )
#     print("\n MySQL Connection Successful!")
#     conn.close()
# except mysql.connector.Error as err:
#     print("\n MySQL Connection Failed!", err)

# #  3. Check Flask Server
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return " Flask Server is Running!"

# if __name__ == '__main__':
#     print("\n Starting Flask Server... Open 'http://127.0.0.1:5000' in your browser.")
#     app.run(debug=True, use_reloader=False)

# #  4. Generate a Matplotlib Pop-Up Plot
# sns.set_theme(style="darkgrid")
# data = np.random.rand(100, 2)
# df = pd.DataFrame(data, columns=["X", "Y"])
# plt.figure(figsize=(6, 4))
# sns.scatterplot(x="X", y="Y", data=df)
# plt.title(" Matplotlib & Seaborn Test Plot")
# plt.show()


# # - Integrating TensorFlow models into Flask
# # - Building an interactive data dashboard with Dash. 
# # - Structuring a Flask-based full-stack application
