# from flask import Flask, render_template, request, redirect, session, url_for
# import csv
# import os
# from werkzeug.security import generate_password_hash, check_password_hash
# from datetime import datetime

# app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'  # Keiskite į saugų raktą

# # Slaptažodžio hash'as (testavimui - 'password')
# CORRECT_PASSWORD_HASH = generate_password_hash('password')

# # CSV failo inicializavimas su reikiamais stulpeliais
# def init_csv():
#     if not os.path.exists('purchases.csv'):
#         with open('purchases.csv', 'w', newline='', encoding='utf-8') as file:
#             writer = csv.writer(file)
#             writer.writerow([
#                 'Invoice', 'StockCode', 'Description', 'Quantity',
#                 'InvoiceDate', 'Price', 'Customer_ID', 'Country', 'Revenue'
#             ])

# @app.route('/')
# def home():
#     if not session.get('logged_in'):
#         return redirect(url_for('login'))
#     return redirect(url_for('index'))

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         password = request.form['password']
#         if check_password_hash(CORRECT_PASSWORD_HASH, password):
#             session['logged_in'] = True
#             return redirect(url_for('index'))
#         else:
#             return render_template('login.html', error='Neteisingas slaptažodis')
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.pop('logged_in', None)
#     return redirect(url_for('login'))

# @app.route('/index')
# def index():
#     if not session.get('logged_in'):
#         return redirect(url_for('login'))
#     return render_template('index.html')

# @app.route('/add_purchase', methods=['POST'])
# def add_purchase():
#     if not session.get('logged_in'):
#         return redirect(url_for('login'))
    
#     # Gauti formos duomenis
#     invoice = request.form['invoice']
#     stock_code = request.form['stock_code']
#     description = request.form['description']
#     quantity = int(request.form['quantity'])
#     invoice_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     price = float(request.form['price'])
#     customer_id = request.form['customer_id']
#     country = request.form['country']
#     revenue = quantity * price  # Apskaičiuojamos pajamos
    
#     # Įrašyti į CSV failą
#     with open('purchases.csv', 'a', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow([
#             invoice, stock_code, description, quantity,
#             invoice_date, price, customer_id, country, revenue
#         ])
    
#     return redirect(url_for('index'))




# app = Flask(__name__)

# @app.route('/dashboard')
# def dashboard():
#     purchases = []
#     with open('purchases.csv', 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             purchases.append(row)
#     return render_template('dashboard.html', purchases=purchases)



# if __name__ == '__main__':
#     init_csv()
#     app.run(debug=True)





from flask import Flask, render_template, request, redirect, session, url_for
import csv
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change to a secure random key

# Password configuration
CORRECT_PASSWORD_HASH = generate_password_hash('password')  # Change this password

# Initialize CSV with required columns
def init_csv():
    if not os.path.exists('purchases.csv'):
        with open('purchases.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Invoice', 'StockCode', 'Description', 'Quantity',
                'InvoiceDate', 'Price', 'Customer_ID', 'Country', 'Revenue'
            ])

# # Authentication routes
# @app.route('/')
# def home():
#     if not session.get('logged_in'):
#         return redirect(url_for('login'))
#     return redirect(url_for('index'))


@app.route('/', methods=['GET', 'POST'])  # Add POST here
def home():
    if request.method == 'POST':
        password = request.form.get('password')
        if password and check_password_hash(CORRECT_PASSWORD_HASH, password):
            session['logged_in'] = True
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid password')
    return render_template('login.html')


# @app.route('/')
# def home():
#     # Changed from redirect(url_for('login')) to render_template directly
#     return render_template('login.html')


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

# Purchase management routes
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

# Dashboard route
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    purchases = []
    if os.path.exists('purchases.csv'):
        with open('purchases.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            purchases = list(reader)
    return render_template('dashboard.html', purchases=purchases)

if __name__ == '__main__':
    init_csv()
    app.run(debug=True)