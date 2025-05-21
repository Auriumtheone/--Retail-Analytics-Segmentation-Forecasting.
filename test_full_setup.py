# import tensorflow as tf
# import numpy as np
# import flask
# import mysql.connector
# import fastapi
# import uvicorn
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from flask import Flask, request, jsonify
# from fastapi import FastAPI

# #  1. TensorFlow Model Check
# print("\n TensorFlow Version:", tf.__version__)
# model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
# model.compile(optimizer="sgd", loss="mse")
# sample_input = np.array([[5.0]])
# prediction = model.predict(sample_input)
# print(" TensorFlow Model Prediction:", prediction)

# #  2. MySQL Connection Test
# try:
#     conn = mysql.connector.connect(
#         host="localhost",
#         user="ORACLETM",      # Change to your MySQL username
#         password="akle7",  # Change to your MySQL password
#         database="test_db"  # Ensure the database exists
#     )
#     print(" MySQL Connection Successful!")
#     conn.close()
# except mysql.connector.Error as err:
#     print(" MySQL Connection Failed!", err)

# #  3. Flask API with TensorFlow Model
# flask_app = Flask(__name__)

# @flask_app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json["input"]
#     pred = model.predict(np.array([[data]]))
#     return jsonify({"prediction": float(pred[0][0])})

# def run_flask():
#     print("\n Starting Flask API... Visit http://127.0.0.1:5000/predict")
#     flask_app.run(debug=True, use_reloader=False)

# #  4. FastAPI Alternative for ML Serving
# fastapi_app = FastAPI()

# @fastapi_app.post("/predict")
# async def fast_predict(input_data: dict):
#     data = input_data["input"]
#     pred = model.predict(np.array([[data]]))
#     return {"prediction": float(pred[0][0])}

# def run_fastapi():
#     print("\n Starting FastAPI... Visit http://127.0.0.1:8000/predict")
#     uvicorn.run(fastapi_app, host="127.0.0.1", port=8000)

# #  5. Dash Data Visualization
# dash_app = dash.Dash(__name__)

# dash_app.layout = html.Div([
#     html.H1("TensorFlow Predictions Dashboard"),
#     dcc.Graph(
#         figure={
#             "data": [{"x": [1, 2, 3, 4, 5], "y": [model.predict(np.array([[i]]))[0][0] for i in range(1, 6)], "type": "line", "name": "Predictions"}],
#             "layout": {"title": "Model Prediction Results"}
#         }
#     )
# ])

# def run_dash():
#     print("\n Starting Dash Dashboard... Open http://127.0.0.1:8050")
#     dash_app.run_server(debug=True, use_reloader=False)

# #  Running all components in parallel
# import threading

# flask_thread = threading.Thread(target=run_flask)
# fastapi_thread = threading.Thread(target=run_fastapi)
# dash_thread = threading.Thread(target=run_dash)

# flask_thread.start()
# fastapi_thread.start()
# dash_thread.start()

# flask_thread.join()
# fastapi_thread.join()
# dash_thread.join()



#################################



#########################################

import tensorflow as tf
import numpy as np
import flask
import mysql.connector
import dash
import matplotlib.pyplot as plt
import seaborn as sns
from dash import dcc, html
from flask import Flask, request, jsonify
import threading

#  1. TensorFlow Model Check
print("\n🔹 TensorFlow Version:", tf.__version__)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="sgd", loss="mse")

sample_input = np.array([[5.0]])
prediction = model.predict(sample_input)
print(" TensorFlow Model Prediction:", prediction)

#  2. MySQL Connection Test
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="ORACLETM",  # Change to your MySQL username
        password="akle7",  # Change to your MySQL password
        database="test_db"  # Ensure the database exists
    )
    print(" MySQL Connection Successful!")
    conn.close()
except mysql.connector.Error as err:
    print(" MySQL Connection Failed!", err)

#  3. Matplotlib & Seaborn Visualization (RUN THIS FIRST)
sns.set_theme(style="darkgrid")
data = np.arange(1, 6).reshape(-1, 1)
predictions = [model.predict(np.array([[i]]))[0][0] for i in range(1, 6)]

plt.figure(figsize=(8, 5))
sns.lineplot(x=data.flatten(), y=predictions, marker="o", label="Predictions")
plt.xlabel("Input Values")
plt.ylabel("Predicted Output")
plt.title("TensorFlow Model Predictions")
plt.show()  #  Run this FIRST before Flask/Dash!

#  4. Flask API with TensorFlow Model
flask_app = Flask(__name__)

@flask_app.route('/predict', methods=['POST'])
def predict():
    data = request.json["input"]
    pred = model.predict(np.array([[data]]))
    return jsonify({"prediction": float(pred[0][0])})

def run_flask():
    print("\n Starting Flask API... Visit http://127.0.0.1:5000/predict")
    flask_app.run(debug=True, use_reloader=False)

#  5. Dash Data Visualization
dash_app = dash.Dash(__name__)

dash_app.layout = html.Div([
    html.H1("TensorFlow Predictions Dashboard"),
    dcc.Graph(
        figure={
            "data": [{"x": [1, 2, 3, 4, 5], "y": predictions, "type": "line", "name": "Predictions"}],
            "layout": {"title": "Model Prediction Results"}
        }
    )
])

def run_dash():
    print("\n Starting Dash Dashboard... Open http://127.0.0.1:8050")
dash_app.run(debug=True)

#  Running Flask & Dash in Parallel (Matplotlib Runs First)
flask_thread = threading.Thread(target=run_flask)
dash_thread = threading.Thread(target=run_dash)

flask_thread.start()
dash_thread.start()

flask_thread.join()
dash_thread.join()



# - Interactive Dash elements (sliders, dropdowns, dynamic updates)
# - More Flask endpoints (database queries, authentication, etc.)
# - Improving model inference (handling batch inputs, validation)


###############################################################################################################

#  Modify Your Dash Layout to include a slider

from dash.dependencies import Input, Output

dash_app.layout = html.Div([
    html.H1("TensorFlow Predictions Dashboard"),
    dcc.Slider(
        id="input-slider",
        min=1,
        max=10,
        step=1,
        value=5
    ),
    html.Div(id="output-prediction"),
    dcc.Graph(id="prediction-graph")
])

@dash_app.callback(
    [Output("output-prediction", "children"), Output("prediction-graph", "figure")],
    [Input("input-slider", "value")]
)
def update_prediction(input_value):
    pred = model.predict(np.array([[input_value]]))[0][0]
    figure = {
        "data": [{"x": list(range(1, 11)), "y": [model.predict(np.array([[i]]))[0][0] for i in range(1, 11)], "type": "line", "name": "Predictions"}],
        "layout": {"title": "Updated Model Predictions"}
    }
    return f"Prediction: {pred}", figure


# Now, users can slide to select an input value and instantly see predictions!



##########################################################################################################
#      Modify Flask to include a MySQL data fetch endpoint

@flask_app.route('/fetch_data', methods=['GET'])
def fetch_data():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="ORACLETM",
            password="akle7",
            database="test_db"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions")  # Modify for your table
        results = cursor.fetchall()
        conn.close()
        return jsonify({"data": results})
    except mysql.connector.Error as err:
        return jsonify({"error": str(err)})
    


    #####################################################################################

#    Add Simple User Authentication




from flask import request

@flask_app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    if username == "admin" and password == "password123":  # Replace with actual validation logic
        return jsonify({"message": "Login successful"})
    return jsonify({"message": "Invalid credentials"})





#  Pridėti SQLAlchemy modelius (models.py) ir naudoti ORM užklausoms
# ✔ Pridėti daugiau Flask API endpointų (import.py, train.py, test.py)
# ✔ Įgalinti nuotraukų duomenu naudojant Flask-Uploads arba FastAPI
# ✔ Pridėti papildomus ML modelius (models/knn.py, models/cnn.py, models/rnn.py)
# ✔ Pridėti duomenų normalizavimą (data_processing.py)
# ✔ Sukurti naudotojo sąsajos formą (dash_ui.py)


########################################################################################################################################
########################################################################################################################################

# import os
# from dotenv import load_dotenv
# import tensorflow as tf
# import numpy as np
# from flask import Flask, request, jsonify
# from fastapi import FastAPI
# import mysql.connector
# import dash
# from dash import dcc, html
# import uvicorn
# import threading
# import logging

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # 1. Enhanced TensorFlow Model
# def setup_model():
#     logger.info("\nTensorFlow Version: %s", tf.__version__)
    
#     # More meaningful test model
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
#         tf.keras.layers.Dense(1)
#     ])
    
#     model.compile(optimizer='adam', loss='mse')
    
#     # Train with sample data
#     X = np.arange(0, 10, 0.1)
#     y = X * 2 + np.random.normal(scale=0.5, size=X.shape)
#     model.fit(X, y, epochs=10, verbose=0)
    
#     return model

# model = setup_model()

# # 2. Secure MySQL Test
# def test_mysql():
#     try:
#         conn = mysql.connector.connect(
#             host=os.getenv("DB_HOST", "localhost"),
#             user=os.getenv("DB_USER"),
#             password=os.getenv("DB_PASSWORD"),
#             database=os.getenv("DB_NAME", "test_db")
#         )
        
#         cursor = conn.cursor()
        
#         # Test table operations
#         cursor.execute("CREATE TABLE IF NOT EXISTS test (id INT AUTO_INCREMENT PRIMARY KEY, value FLOAT)")
#         cursor.execute("INSERT INTO test (value) VALUES (%s)", (model.predict([[5]])[0][0],))
#         conn.commit()
        
#         cursor.execute("SELECT * FROM test")
#         logger.info("MySQL Test Successful. Records: %s", cursor.fetchall())
        
#         cursor.close()
#         conn.close()
#     except Exception as e:
#         logger.error("MySQL Test Failed: %s", e)

# test_mysql()

# # 3. Flask API with Error Handling
# flask_app = Flask(__name__)

# @flask_app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json["input"]
#         pred = model.predict(np.array([[float(data)]]))
#         return jsonify({"prediction": float(pred[0][0])})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# def run_flask():
#     logger.info("\nStarting Flask API on port 5000...")
#     flask_app.run(port=5000, debug=False)

# # 4. FastAPI with Pydantic Validation
# fastapi_app = FastAPI()

# from pydantic import BaseModel

# class PredictionRequest(BaseModel):
#     input: float

# @fastapi_app.post("/predict")
# async def fast_predict(request: PredictionRequest):
#     try:
#         pred = model.predict(np.array([[request.input]]))
#         return {"prediction": float(pred[0][0])}
#     except Exception as e:
#         return {"error": str(e)}

# def run_fastapi():
#     logger.info("\nStarting FastAPI on port 8000...")
#     uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# # 5. Interactive Dash Dashboard
# dash_app = dash.Dash(__name__)

# dash_app.layout = html.Div([
#     html.H1("Enhanced Model Dashboard"),
#     dcc.Input(id='input-value', type='number', value=5),
#     html.Button('Predict', id='predict-button'),
#     dcc.Graph(id='prediction-graph'),
#     html.Div(id='live-prediction')
# ])

# # Add callbacks for interactivity
# from dash.dependencies import Input, Output

# @dash_app.callback(
#     [Output('prediction-graph', 'figure'),
#      Output('live-prediction', 'children')],
#     [Input('predict-button', 'n_clicks')],
#     [dash.dependencies.State('input-value', 'value')]
# )
# def update_output(n_clicks, value):
#     if value is None:
#         return dash.no_update
    
#     x_values = np.linspace(0, 10, 50)
#     y_values = model.predict(x_values.reshape(-1, 1)).flatten()
    
#     current_pred = model.predict([[value]])[0][0]
    
#     return (
#         {
#             'data': [{
#                 'x': x_values,
#                 'y': y_values,
#                 'type': 'line',
#                 'name': 'Model'
#             }],
#             'layout': {
#                 'title': f'Model Predictions (Current: {value:.2f} → {current_pred:.2f})'
#             }
#         },
#         f"Prediction for {value}: {current_pred:.4f}"
#     )

# def run_dash():
#     logger.info("\nStarting Dash on port 8050...")
#     dash_app.run_server(port=8050, debug=False)

# # Main execution
# if __name__ == "__main__":
#     # Create .env file if it doesn't exist
#     if not os.path.exists('.env'):
#         with open('.env', 'w') as f:
#             f.write("DB_HOST=localhost\nDB_USER=youruser\nDB_PASSWORD=yourpass\nDB_NAME=test_db")
    
#     # Start services with port conflict handling
#     try:
#         threading.Thread(target=run_flask, daemon=True).start()
#         threading.Thread(target=run_fastapi, daemon=True).start()
#         run_dash()  # Run Dash in main thread
#     except Exception as e:
#         logger.error("Failed to start services: %s", e)



# # tensorflow>=2.0
# # flask
# # mysql-connector-python
# # python-dotenv
# # fastapi
# # uvicorn
# # dash
# # dash-core-components
# # dash-html-components
# # dash-renderer





#########################################################################################################################
#######################################################################################################################



# import tensorflow as tf
# import numpy as np
# import flask
# import mysql.connector
# import dash
# import matplotlib.pyplot as plt
# import seaborn as sns
# from dash import dcc, html
# from flask import Flask, request, jsonify
# import threading
# import logging
# from dash.dependencies import Input, Output  # THIS WAS MISSING

# # Configure logging to prevent duplicates
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.propagate = False  # Prevent duplicate logs

# # 1. TensorFlow Model Check
# logger.info("\n TensorFlow Version: %s", tf.__version__)
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(1,)),
#     tf.keras.layers.Dense(1)
# ])
# model.compile(optimizer="sgd", loss="mse")

# sample_input = np.array([[5.0]])
# prediction = model.predict(sample_input)
# logger.info("TensorFlow Model Prediction: %s", prediction)

# # 2. MySQL Connection Test
# try:
#     conn = mysql.connector.connect(
#         host="localhost",
#         user="ORACLETM",
#         password="akle7",
#         database="test_db"
#     )
#     logger.info("MySQL Connection Successful!")
#     conn.close()
# except mysql.connector.Error as err:
#     logger.error("MySQL Connection Failed! %s", err)

# # 3. Matplotlib & Seaborn Visualization
# def show_plot():
#     sns.set_theme(style="darkgrid")
#     data = np.arange(1, 6).reshape(-1, 1)
#     predictions = [model.predict(np.array([[i]]))[0][0] for i in range(1, 6)]
    
#     plt.figure(figsize=(8, 5))
#     sns.lineplot(x=data.flatten(), y=predictions, marker="o", label="Predictions")
#     plt.xlabel("Input Values")
#     plt.ylabel("Predicted Output")
#     plt.title("TensorFlow Model Predictions")
#     plt.show()

# # 4. Flask API
# flask_app = Flask(__name__)

# @flask_app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json["input"]
#     pred = model.predict(np.array([[data]]))
#     return jsonify({"prediction": float(pred[0][0])})

# def run_flask():
#     logger.info("\nStarting Flask API... Visit http://127.0.0.1:5000/predict")
#     flask_app.run(debug=False, use_reloader=False)

# # 5. Dash App
# dash_app = dash.Dash(__name__)

# dash_app.layout = html.Div([
#     html.H1("TensorFlow Predictions Dashboard"),
#     dcc.Slider(
#         id="input-slider",
#         min=1,
#         max=10,
#         step=1,
#         value=5
#     ),
#     html.Div(id="output-prediction"),
#     dcc.Graph(id="prediction-graph")
# ])

# @dash_app.callback(
#     [Output("output-prediction", "children"), Output("prediction-graph", "figure")],
#     [Input("input-slider", "value")]
# )
# def update_prediction(input_value):
#     pred = model.predict(np.array([[input_value]]))[0][0]
#     figure = {
#         "data": [{"x": list(range(1, 11)), "y": [model.predict(np.array([[i]]))[0][0] for i in range(1, 11)], "type": "line", "name": "Predictions"}],
#         "layout": {"title": f"Prediction for {input_value}: {pred:.2f}"}
#     }
#     return f"Prediction: {pred:.4f}", figure

# def run_dash():
#     logger.info("\nStarting Dash Dashboard... Open http://127.0.0.1:8050")
#     dash_app.run_server(debug=False)

# # Main execution flow
# if __name__ == "__main__":
#     # Run visualization first
#     show_plot()
    
#     # Start services
#     flask_thread = threading.Thread(target=run_flask, daemon=True)
#     dash_thread = threading.Thread(target=run_dash, daemon=True)
    
#     flask_thread.start()
#     dash_thread.start()
    
#     try:
#         while True:  # Keep main thread alive
#             pass
#     except KeyboardInterrupt:
#         logger.info("Shutting down servers...")







#########################################################################################################
#########################################################################################################



# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

# import tensorflow as tf
# import numpy as np
# import mysql.connector
# import dash
# from dash import dcc, html, Input, Output
# import flask
# from flask import Flask, request, jsonify
# import threading
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.propagate = False

# # 1. TensorFlow Model Check
# logger.info("\n🔹 TensorFlow Version: %s", tf.__version__)
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(1,)),
#     tf.keras.layers.Dense(1)
# ])
# model.compile(optimizer="sgd", loss="mse")

# sample_input = np.array([[5.0]])
# prediction = model.predict(sample_input, verbose=0)
# logger.info("TensorFlow Model Prediction: %s", prediction)

# # 2. MySQL Connection Test
# try:
#     conn = mysql.connector.connect(
#         host="localhost",
#         user="ORACLETM",
#         password="akle7",
#         database="test_db"
#     )
#     logger.info("MySQL Connection Successful!")
#     conn.close()
# except mysql.connector.Error as err:
#     logger.error("MySQL Connection Failed! %s", err)

# # 3. Matplotlib & Seaborn Visualization
# def show_plot():
#     sns.set_theme(style="darkgrid")
#     data = np.arange(1, 6).reshape(-1, 1)
#     predictions = [model.predict(np.array([[i]]), verbose=0)[0][0] for i in range(1, 6)]
    
#     plt.figure(figsize=(8, 5))
#     sns.lineplot(x=data.flatten(), y=predictions, marker="o", label="Predictions")
#     plt.xlabel("Input Values")
#     plt.ylabel("Predicted Output")
#     plt.title("TensorFlow Model Predictions")
#     plt.show()

# # 4. Flask API
# flask_app = Flask(__name__)

# @flask_app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json["input"]
#     pred = model.predict(np.array([[data]]), verbose=0)
#     return jsonify({"prediction": float(pred[0][0])})

# def run_flask():
#     logger.info("\nStarting Flask API... Visit http://127.0.0.1:5000/predict")
#     flask_app.run(debug=False, use_reloader=False)

# # 5. Dash App
# dash_app = dash.Dash(__name__, suppress_callback_exceptions=True)

# dash_app.layout = html.Div([
#     html.H1("TensorFlow Predictions Dashboard"),
#     dcc.Slider(
#         id="input-slider",
#         min=1,
#         max=10,
#         step=1,
#         value=5,
#         marks={i: str(i) for i in range(1, 11)}
#     ),
#     html.Div(id="output-prediction"),
#     dcc.Graph(id="prediction-graph")
# ])

# @dash_app.callback(
#     [Output("output-prediction", "children"), 
#      Output("prediction-graph", "figure")],
#     [Input("input-slider", "value")]
# )
# def update_prediction(input_value):
#     pred = model.predict(np.array([[input_value]]), verbose=0)[0][0]
#     figure = {
#         "data": [{
#             "x": list(range(1, 11)), 
#             "y": [model.predict(np.array([[i]]), verbose=0)[0][0] for i in range(1, 11)], 
#             "type": "line", 
#             "name": "Predictions"
#         }],
#         "layout": {"title": f"Prediction for {input_value}: {pred:.2f}"}
#     }
#     return f"Prediction: {pred:.4f}", figure

# def run_dash():
#     logger.info("\nStarting Dash Dashboard... Open http://127.0.0.1:8050")
#     dash_app.run(port=8050, debug=False)

# # Main execution
# if __name__ == "__main__":
#     show_plot()  # Show plot first
    
#     # Start services
#     flask_thread = threading.Thread(target=run_flask, daemon=True)
#     dash_thread = threading.Thread(target=run_dash, daemon=True)
    
#     flask_thread.start()
#     dash_thread.start()
    
#     try:
#         while True:  # Keep main thread alive
#             pass
#     except KeyboardInterrupt:
#         logger.info("Shutting down servers...")




##############################################################


#############################################################


# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors
# os.environ['WERKZEUG_RUN_MAIN'] = 'true'

# import tensorflow as tf
# import numpy as np
# import mysql.connector
# import dash
# from dash import dcc, html, Input, Output
# from flask import Flask, request, jsonify
# import threading
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Configure clean logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(message)s',
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)
# logger.propagate = False

# # Suppress other loggers
# logging.getLogger('werkzeug').setLevel(logging.WARNING)
# logging.getLogger('dash').setLevel(logging.WARNING)
# logging.getLogger('matplotlib').setLevel(logging.WARNING)

# # ==================== 1. TensorFlow Model with Visible Training ====================
# logger.info("\n TensorFlow Model Training")
# X_train = np.arange(0, 10, 0.1).reshape(-1, 1)
# y_train = X_train * 2 + np.random.normal(scale=0.5, size=X_train.shape)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(optimizer='adam', loss='mse')

# # Train with progress display
# history = model.fit(
#     X_train, 
#     y_train,
#     epochs=10,
#     batch_size=8,
#     verbose=1  # Show progress bars
# )

# logger.info(" Training complete - Final loss: %.4f", history.history['loss'][-1])

# # ==================== 2. MySQL Connection Test ====================
# try:
#     conn = mysql.connector.connect(
#         host="localhost",
#         user="ORACLETM",
#         password="akle7",
#         database="test_db"
#     )
#     cursor = conn.cursor()
#     cursor.execute("CREATE TABLE IF NOT EXISTS predictions (id INT AUTO_INCREMENT PRIMARY KEY, value FLOAT, prediction FLOAT)")
    
#     # Store sample predictions
#     sample_values = [1, 2, 3, 4, 5]
#     for val in sample_values:
#         pred = model.predict([[val]], verbose=0)[0][0]
#         cursor.execute("INSERT INTO predictions (value, prediction) VALUES (%s, %s)", (val, float(pred)))
    
#     conn.commit()
#     cursor.close()
#     conn.close()
#     logger.info(" MySQL: Stored %d predictions", len(sample_values))
# except Exception as e:
#     logger.error(" MySQL Error: %s", str(e))

# # ==================== 3. Visualization ====================
# def show_plot():
#     plt.figure(figsize=(10, 6))
#     x_test = np.linspace(0, 10, 50)
#     y_pred = model.predict(x_test.reshape(-1, 1), verbose=0)
    
#     sns.scatterplot(x=X_train.flatten(), y=y_train.flatten(), label='Training Data')
#     sns.lineplot(x=x_test, y=y_pred.flatten(), color='red', label='Model Predictions')
#     plt.title("Model Performance")
#     plt.xlabel("Input")
#     plt.ylabel("Output")
#     plt.legend()
#     plt.show()

# # ==================== 4. Flask API ====================
# flask_app = Flask(__name__)
# flask_app.logger.disabled = True

# @flask_app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = float(request.json["input"])
#         pred = model.predict([[data]], verbose=0)[0][0]
#         return jsonify({
#             "status": "success",
#             "input": data,
#             "prediction": float(pred)
#         })
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 400

# def run_flask():
#     logger.info("\n Flask API running at http://127.0.0.1:5000")
#     logger.info("POST /predict with JSON {'input': number}")
#     flask_app.run(port=5000, debug=False, use_reloader=False)

# # ==================== 5. Dash Dashboard ====================
# dash_app = dash.Dash(__name__, suppress_callback_exceptions=True)
# dash_app.logger.setLevel(logging.WARNING)

# dash_app.layout = html.Div([
#     html.H1("Live Model Dashboard", style={'textAlign': 'center'}),
#     dcc.Graph(id='live-graph'),
#     dcc.Slider(
#         id='value-slider',
#         min=0,
#         max=10,
#         step=0.1,
#         value=5,
#         marks={i: str(i) for i in range(11)},
#         tooltip={"placement": "bottom"}
#     ),
#     html.Div(id='prediction-output', style={
#         'marginTop': 20,
#         'fontSize': 20,
#         'textAlign': 'center'
#     })
# ])

# @dash_app.callback(
#     [Output('live-graph', 'figure'),
#      Output('prediction-output', 'children')],
#     [Input('value-slider', 'value')]
# )
# def update_dashboard(input_value):
#     x_range = np.linspace(0, 10, 100)
#     y_pred = model.predict(x_range.reshape(-1, 1), verbose=0).flatten()
#     current_pred = model.predict([[input_value]], verbose=0)[0][0]
    
#     fig = {
#         'data': [
#             {'x': X_train.flatten(), 'y': y_train.flatten(), 'mode': 'markers', 'name': 'Training Data'},
#             {'x': x_range, 'y': y_pred, 'mode': 'lines', 'name': 'Predictions', 'line': {'color': 'red'}},
#             {'x': [input_value], 'y': [current_pred], 'mode': 'markers', 'marker': {'size': 12, 'color': 'green'}, 'name': 'Current'}
#         ],
#         'layout': {
#             'title': f'Model Prediction (Input: {input_value:.1f} → Output: {current_pred:.2f})',
#             'xaxis': {'title': 'Input Value'},
#             'yaxis': {'title': 'Predicted Output'}
#         }
#     }
    
#     text = f"Input: {input_value:.1f} → Prediction: {current_pred:.4f}"
    
#     return fig, text

# def run_dash():
#     logger.info("\n Dashboard running at http://127.0.0.1:8050")
#     dash_app.run_server(port=8050, debug=False)

# # ==================== Main Execution ====================
# if __name__ == "__main__":
#     show_plot()
    
#     # Start servers in threads
#     flask_thread = threading.Thread(target=run_flask, daemon=True)
#     dash_thread = threading.Thread(target=run_dash, daemon=True)
    
#     flask_thread.start()
#     dash_thread.start()
    
#     try:
#         while True:
#             pass  # Keep main thread alive
#     except KeyboardInterrupt:
#         logger.info("\n Servers stopped by user")