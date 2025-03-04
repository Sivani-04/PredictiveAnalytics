import json
from flask import Flask,jsonify, request
import psycopg2 # type: ignore # pip install psycopg2
from psycopg2 import sql # type: ignore
from flask_bcrypt import Bcrypt # type: ignore # pip install 
import jwt # pip install pyjwt
from datetime import datetime, timedelta

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
             date DATE NOT NULL,
             predicted_sales INTEGER NOT NULL
        )
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
             inventory_id SERIAL PRIMARY KEY,
             product_name TEXT NOT NULL,
             stock_level INTEGER NOT NULL,
             reorder_point INTEGER NOT NULL,
             optimal_stock INTEGER NOT NULL
        );
    """)
    connection.commit()
    cursor.close() 
    connection.close()

create_users_table_if_not_exists()
create_sales_table_if_not_exists()
create_predictions_table_if_not_exists()
create_inventory_table_if_not_exists()

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

if __name__ == "__main__":
    app.run(debug=True)