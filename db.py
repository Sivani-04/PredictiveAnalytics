MODEL_FILE = "sales_model.pkl"
MODEL_FILE = "sales_forecast.pkl"
import json
from flask import Flask,jsonify, request, send_file
import psycopg2 # type: ignore # pip install psycopg2
from psycopg2 import sql # type: ignore
from flask_bcrypt import Bcrypt # type: ignore # pip install 
import jwt # type: ignore # pip install pyjwt
from datetime import datetime, timedelta
import pandas as pd # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import numpy as np # type: ignore
import pickle
import logging
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import io

app = Flask(__name__)

#Database connection configuaration
DB_HOST = 'localhost'
DB_NAME = 'postgres'
DB_USER = 'postgres'
DB_PASSWORD = 'POSTGRESQL'

# Your secret key to sign JWT tokens
SECRET_KEY = "this is a secret key this is a secret keyyyy!!!!"

# Function to get a database connection
def get_db_connection():
    connection = psycopg2.connect(
        host = DB_HOST,
        database = DB_NAME, 
        user = DB_USER,
        password = DB_PASSWORD
    )
    return connection

# Create the 'users' table if it doesn't exist
def create_users_table_if_not_exists():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            email_id TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    connection.commit()
    cursor.close()
    connection.close()

# Create the 'sales data' table if it doesn't exist
def create_sales_table_if_not_exists():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
             sales_id SERIAL PRIMARY KEY,
             date DATE NOT NULL,
             sales INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    connection.commit()
    cursor.close()
    connection.close()

# Create the 'predictions' table if it doesn't exist
def create_predictions_table_if_not_exists():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
             prediction_id SERIAL PRIMARY KEY,
             date DATE UNIQUE,
             predicted_sales FLOAT
        );
    """)
    connection.commit()
    cursor.close()
    connection.close()

# Create the 'inventory' table if it doesn't 
def create_inventory_table_if_not_exists():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            id SERIAL PRIMARY KEY,
            days INT NOT NULL,
            safety_stock DECIMAL(10,2) NOT NULL,
            optimized_inventory DECIMAL(10,2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    connection.commit()
    cursor.close() 
    connection.close()

# Create the 'pricing' table if it doesn't 
def create_pricing_table_if_not_exists():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pricing (
             id SERIAL PRIMARY KEY,
             product_id INTEGER NOT NULL,
             current_price DECIMAL(10,2) NOT NULL,
             optimized_price DECIMAL(10,2) NOT NULL,
             demand_elasticity DECIMAL(10,2) NOT NULL,
             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    connection.commit()
    cursor.close() 
    connection.close()

create_users_table_if_not_exists()
create_sales_table_if_not_exists()
create_predictions_table_if_not_exists()
create_inventory_table_if_not_exists()
create_pricing_table_if_not_exists()

bcrypt = Bcrypt(app)

def encode_password(password):
    return bcrypt.generate_password_hash(password).decode('utf-8')

def check_password(hashed_password,password):
    return bcrypt.check_password_hash(hashed_password, password)

def decode_token(jwt_token):
    try:
        decoded_token_payload = jwt.decode(jwt_token, SECRET_KEY, algorithms=["HS256"])
        return decoded_token_payload
    except jwt.ExpiredSignatureError:
        return jsonify({"message": "Token has expired!"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"message": "Invalid token!"}), 401

@app.route('/register-users', methods=['POST'])
def register_user():
    data = request.json
    email_id = request.json['email_id']
    username = request.json['username']
    password = request.json['password']
    hashed_password = encode_password(password)
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
            INSERT INTO users (email_id, username, password) VALUES (%s, %s, %s);
        """, (email_id, username, hashed_password))
    connection.commit()
    cursor.close()
    connection.close()
    return jsonify ({"message": "User registered successfully"}),  201

@app.route('/login', methods=['POST'])
def login_user():
    username = request.json['username']
    password = request.json['password']
    connection = get_db_connection()
    cursor = connection.cursor()
    # Check if the username exists
    cursor.execute("SELECT * FROM users WHERE username = %s;", (username,))
    user = cursor.fetchone()
    # If the user does not exist
    if user is None:
        return jsonify({"message": "Invalid username or password."}), 401
    stored_hashed_password = user[3]
    # Compare the stored hashed password with the provided password
    if not check_password(stored_hashed_password, password):
        return jsonify({"message": "Invalid username or password."}), 401
    payload = {
        'username': username,
        'user_id': user[0],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # Token expiration time
    }
    # Generate the token
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    cursor.close()
    connection.close()
    return jsonify({
        "message": "Login successful.",
        "token": token
    }), 200

@app.route('/add-sales', methods=['POST'])
def add_sales():
    data = request.json
    date = data['date']
    sales = data['sales']

    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO sales (date, sales) VALUES (%s, %s);", (date, sales))
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({"message": "Sales data added successfully"}), 201

@app.route('/get-sales', methods=['GET'])
def get_sales():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if not start_date or not end_date:
        return jsonify({"error": "Missing start_date or end_date"}), 400

    try:
        # Convert input dates
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT date, sales FROM sales WHERE date BETWEEN %s AND %s;", (start_date, end_date))
        sales = cursor.fetchall()
        cursor.close()
        conn.close()

        if not sales:
            return jsonify({"message": "No sales data found for the given date range"}), 404

        # Format the response to return only "YYYY-MM-DD"
        return jsonify([
            {"date": row[0].strftime("%Y-%m-%d"), "sales": row[1]} for row in sales
        ])

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/train-sales-model', methods=['POST'])
def train_sales_model():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT date, sales FROM sales ORDER BY date ASC;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({"error": "No sales data available"}), 404

        # Convert DB data to DataFrame
        df = pd.DataFrame(rows, columns=['date', 'sales'])
        df['date'] = pd.to_datetime(df['date'])
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

        # Prepare training data
        X = df[['days_since_start']]
        y = df['sales']

        # Train the model
        model = LinearRegression()
        model.fit(X, y)

        # Save the trained model
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)

        return jsonify({"message": "Sales forecasting model trained successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-sales', methods=['POST'])
def predict_sales():
    try:
        data = request.json
        days = data.get("days", 30)

        # Load trained model
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Get last available sales date
        cursor.execute("SELECT MAX(date) FROM sales;")
        last_date = cursor.fetchone()[0]

        if not last_date:
            return jsonify({"error": "No sales data available"}), 404

        last_date = pd.to_datetime(last_date)
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        future_days_since_start = [(date - last_date).days for date in future_dates]

        # Predict future sales
        predicted_sales = model.predict(np.array(future_days_since_start).reshape(-1, 1))

        predictions = []
        for date, sales in zip(future_dates, predicted_sales):
            formatted_date = date.strftime("%Y-%m-%d")
            predictions.append({"date": formatted_date, "predicted_sales": round(sales, 2)})

            # Fix: Ensure Unique Constraint or Avoid Conflict
            cursor.execute("DELETE FROM predictions WHERE date = %s;", (formatted_date,))
            cursor.execute(
                "INSERT INTO predictions (date, predicted_sales) VALUES (%s, %s);",
                (formatted_date, float(sales))  # Convert np.float64 to float
            )

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify(predictions), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize-pricing', methods=['POST'])
def optimize_pricing():
    try:
        data = request.json
        product_id = data.get("product_id")
        current_price = data.get("current_price")
        demand_elasticity = data.get("demand_elasticity")

        if not product_id or not current_price or not demand_elasticity:
            return jsonify({"error": "Missing required fields"}), 400

        # Price Optimization Formula
        optimized_price = round(current_price * (1 + demand_elasticity), 2)

        # Store optimized price in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pricing (product_id, current_price, optimized_price, demand_elasticity)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """, (product_id, current_price, optimized_price, demand_elasticity))
        record_id = cursor.fetchone()[0]
        conn.commit()

        return jsonify({
            "message": "Price optimization successful",
            "optimized_price": optimized_price,
            "record_id": record_id
        }), 200

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/get-optimized-prices', methods=['GET'])
def get_optimized_prices():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pricing;")
        rows = cursor.fetchall()

        if not rows:
            return jsonify({"message": "No optimized prices found"}), 404

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "product_id": row[1],
                "current_price": float(row[2]),
                "optimized_price": float(row[3]),
                "demand_elasticity": float(row[4]),
                "created_at": row[5].strftime("%Y-%m-%d %H:%M:%S")
            })

        return jsonify(results), 200

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()
 
@app.route('/optimize-inventory', methods=['POST'])
def optimize_inventory():
    try:
        data = request.json
        days = data.get("days", 30)
        safety_stock = data.get("safety_stock", 10)  # Buffer stock
        optimized_inventory = 0.0
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM sales;")
        last_date = cursor.fetchone()[0]

        if not last_date:
            return jsonify({"error": "No sales data available"}), 404

        last_date = pd.to_datetime(last_date)
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        future_days_since_start = [(date - last_date).days for date in future_dates]

        predicted_sales = model.predict(np.array(future_days_since_start).reshape(-1, 1))

        if predicted_sales is None or len(predicted_sales) == 0:
            return jsonify({"error": "Prediction model failed"}), 500

        total_demand = sum(predicted_sales)
        optimized_inventory = total_demand + safety_stock  # Add safety stock buffer

        optimized_inventory = float(optimized_inventory)  # Convert NumPy float to Python float

        # Store in the database
        cursor.execute("""
            INSERT INTO inventory (days, safety_stock, optimized_inventory)
            VALUES (%s, %s, %s) RETURNING id;
        """, (days, safety_stock, optimized_inventory))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"optimized_inventory": optimized_inventory}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/visualize-sales', methods=['GET'])
def visualize_sales():
    try:
        days = int(request.args.get("days", 30))  # Default forecast range: 30 days
        safety_stock = int(request.args.get("safety_stock", 10))  # Default buffer stock

        # Load trained sales forecasting model
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        # Fetch sales data from the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT date, sales FROM sales ORDER BY date ASC;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({"error": "No sales data available"}), 404

        # Convert sales data to DataFrame
        df = pd.DataFrame(rows, columns=['date', 'sales'])
        df['date'] = pd.to_datetime(df['date'])
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

        # Get last date in the dataset
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        future_days_since_start = [(date - df['date'].min()).days for date in future_dates]

        # Predict future sales using the model
        predicted_sales = model.predict(np.array(future_days_since_start).reshape(-1, 1))

        # Calculate optimized inventory (total demand + safety stock)
        total_demand = np.sum(predicted_sales)
        optimized_inventory = total_demand + safety_stock

        # Create visualization
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot actual and predicted sales on primary y-axis
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Sales", color="blue")
        ax1.plot(df['date'], df['sales'], 'bo-', label="Actual Sales")
        ax1.plot(future_dates, predicted_sales, 'r--', label="Predicted Sales")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")

        # Plot optimized inventory on secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Optimized Inventory", color="green")
        ax2.axhline(y=optimized_inventory, color="green", linestyle="dotted", label="Optimized Inventory")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.legend(loc="upper right")

        # Title and Layout
        plt.title("Sales Forecasting & Inventory Optimization")
        plt.grid(True)

        # Save the plot as an image in memory
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plt.close()

        return send_file(img, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)