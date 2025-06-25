import mysql.connector
import os
from dotenv import load_dotenv

def load_config():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        raise FileNotFoundError(f".env file not found at {env_path}")
    
    load_dotenv(env_path)
    
    config = {
        "mysql": {
            "host": os.getenv("DB_HOST"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME")
        }
    }
    
    missing_vars = [key for key, value in config["mysql"].items() if value is None]
    if missing_vars:
        raise ValueError(f"Missing required environment variables in .env: {missing_vars}")
    
    return config

def get_db_connection():
    try:
        config = load_config()
        conn = mysql.connector.connect(
            host=config["mysql"]["host"],
            user=config["mysql"]["user"],
            password=config["mysql"]["password"],
            database=config["mysql"]["database"]
        )
        cursor = conn.cursor()
        cursor.execute("SET SESSION time_zone = '+05:30'")
        cursor.close()
        return conn
    except mysql.connector.Error as e:
        raise Exception(f"Database connection failed: {str(e)}")
    except KeyError as e:
        raise KeyError(f"Missing required key in config: {e}")

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            emp_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            folder_name VARCHAR(100)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_log (
            log_id INT AUTO_INCREMENT PRIMARY KEY,
            emp_id INT,
            status ENUM('IN', 'OUT', 'BREAK_START', 'BREAK_END'),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_break_exceeded BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()