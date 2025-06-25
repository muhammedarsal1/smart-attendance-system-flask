import pandas as pd
import plotly.express as px
import os
import re
import shutil
import cv2
import numpy as np
import base64
from io import BytesIO
from datetime import datetime, timedelta
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file, Response
from db_config import get_db_connection
from recognizer import recognize_and_log, manual_log
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "admin123")

# Define the path for face data
FACE_DATA_PATH = r"C:\Tinos\smart_attendance_systemFlask\face_data"

# Configure Flask to serve face_data directory
app.config['FACE_DATA_PATH'] = FACE_DATA_PATH

# Route to serve files from face_data directory
@app.route('/face_data/<path:filename>')
def serve_face_data(filename):
    return send_file(os.path.join(app.config['FACE_DATA_PATH'], filename))

# Load the face cascade classifier for image capture
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
if face_cascade.empty():
    raise RuntimeError("Error: Could not load face cascade classifier.")

# Initialize recognizer and label dictionary globally
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
label_dict = {}
try:
    if os.path.exists("trainer.yml"):
        recognizer.read("trainer.yml")
        logger.info("Loaded existing trainer.yml")
    if os.path.exists("label_dict.npz"):
        loaded_dict = np.load("label_dict.npz")
        label_dict = {int(k): str(v) for k, v in loaded_dict.items()}
        logger.info("Loaded existing label_dict.npz")
except Exception as e:
    logger.error(f"Could not load model or labels: {str(e)}. Please train the model first.")

# Define the custom filter
@app.template_filter('datetimeformat')
def datetimeformat(value, format_string):
    if value == "now":
        return datetime.now().strftime(format_string)
    try:
        if isinstance(value, str):
            dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        else:
            # Handle NULL timestamp
            if value is None:
                return "N/A"
            dt = value
        return dt.strftime(format_string)
    except (ValueError, TypeError):
        return value if value else "N/A"

def initialize_session():
    if "last_log" not in session:
        session["last_log"] = None
    if "admin_panel_authorized" not in session:
        session["admin_panel_authorized"] = False
    if "employee_name" not in session:
        session["employee_name"] = None
    if "captured_counts" not in session:
        session["captured_counts"] = {"front": 0, "right": 0, "left": 0}
    if "capturing" not in session or not isinstance(session["capturing"], dict):
        session["capturing"] = {"front": False, "right": False, "left": False}
    if "total_captured" not in session:
        session["total_captured"] = 0
    if "selected_angle" not in session:
        session["selected_angle"] = None
    if "training_message" not in session:
        session["training_message"] = None
    if "training_error" not in session:
        session["training_error"] = None
    if "training_progress" not in session:
        session["training_progress"] = None

# Cooldown tracking for video feed
last_recognition = {}
COOLDOWN_SECONDS = 5  # 5 sec

def preprocess_face(face):
    # Resize to a standard size
    face = cv2.resize(face, (100, 100))
    # Apply histogram equalization for better contrast
    face = cv2.equalizeHist(face)
    # Normalize pixel values
    face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)
    return face

def train_model(employee_name=None):
    global recognizer, label_dict
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_dir = FACE_DATA_PATH
    faces = []
    labels = []
    label_dict_local = {}
    current_id = 0

    # Load existing labels if available
    if os.path.exists("label_dict.npz"):
        label_dict_local = np.load("label_dict.npz")
        label_dict_local = {int(k): str(v) for k, v in label_dict_local.items()}
        current_id = max(label_dict_local.keys()) + 1 if label_dict_local else 0
        logger.info(f"Loaded existing labels: {label_dict_local}")

    # Add new employee to label dictionary if provided
    if employee_name and employee_name not in label_dict_local.values():
        label_dict_local[current_id] = employee_name
        logger.info(f"Added new employee {employee_name} with ID {current_id}")
        current_id += 1

    # Collect all face data for training
    for emp_name in os.listdir(face_dir):
        emp_id = None
        for k, v in label_dict_local.items():
            if v == emp_name:
                emp_id = k
                break
        if emp_id is None:
            emp_id = current_id
            label_dict_local[current_id] = emp_name
            logger.info(f"Assigned new ID {emp_id} to employee {emp_name}")
            current_id += 1

        emp_dir = os.path.join(face_dir, emp_name)
        if not os.path.isdir(emp_dir):
            logger.warning(f"Skipping {emp_dir} as it is not a directory")
            continue
        image_files = [f for f in os.listdir(emp_dir) if f.startswith("img") and f.endswith('.jpg')]
        for image_name in image_files:
            image_path = os.path.join(emp_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                face = preprocess_face(image)  # Apply preprocessing
                faces.append(face)
                labels.append(emp_id)
                logger.debug(f"Added image {image_name} for {emp_name} with label {emp_id}")
            else:
                logger.error(f"Failed to load image {image_path}")

    if not faces:
        logger.error("No face data available to train")
        return False, "No face data available to train."

    # Train and save the model
    try:
        logger.info(f"Starting training with {len(faces)} images for {len(label_dict_local)} employees")
        recognizer.train(faces, np.array(labels))
        logger.info("Training completed successfully")
        recognizer.save("trainer.yml")
        logger.info("Saved trainer.yml")

        label_dict_str = {str(k): v for k, v in label_dict_local.items()}
        np.savez("label_dict.npz", **label_dict_str)
        logger.info("Saved label_dict.npz")

        # Verify the files were saved
        if os.path.exists("trainer.yml") and os.path.getsize("trainer.yml") > 0:
            logger.info("Verified trainer.yml exists and is non-empty")
        else:
            logger.error("trainer.yml was not saved or is empty")
            return False, "Failed to save trainer.yml"

        if os.path.exists("label_dict.npz") and os.path.getsize("label_dict.npz") > 0:
            logger.info("Verified label_dict.npz exists and is non-empty")
        else:
            logger.error("label_dict.npz was not saved or is empty")
            return False, "Failed to save label_dict.npz"

        # Verify the model by predicting on the first face (debugging)
        if faces:
            test_label, test_confidence = recognizer.predict(faces[0])
            logger.info(f"Debug: Test prediction - Label: {test_label}, Confidence: {test_confidence}, Expected: {labels[0]}")

        # Update global recognizer and label_dict
        label_dict.clear()
        label_dict.update(label_dict_local)
        recognizer.read("trainer.yml")  # Reload to ensure consistency
        logger.info(f"Updated global label_dict: {label_dict}")
        return True, "Model training completed successfully."
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False, f"Training failed: {str(e)}"

def get_employees():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM employees")
    employees = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return employees

def check_break_timeout():
    conn = get_db_connection()
    cursor = conn.cursor()
    today_date = datetime.now().strftime("%Y-%m-%d")

    query = """
        SELECT al.emp_id, al.timestamp
        FROM attendance_log al
        WHERE al.status = 'BREAK_START'
        AND DATE(al.timestamp) = %s
        AND NOT EXISTS (
            SELECT 1
            FROM attendance_log al2
            WHERE al2.emp_id = al.emp_id
            AND (al2.status = 'BREAK_END' OR al2.status = 'OUT')
            AND DATE(al2.timestamp) = %s
            AND al2.timestamp > al.timestamp
        )
    """
    cursor.execute(query, (today_date, today_date))
    break_starts = cursor.fetchall()

    current_time = datetime.now()
    for emp_id, break_start_time in break_starts:
        if break_start_time is None:
            continue
        time_diff = (current_time - break_start_time).total_seconds() / 3600
        if time_diff >= 2:
            cursor.execute("""
                INSERT INTO attendance_log (emp_id, status, timestamp, is_break_exceeded)
                VALUES (%s, 'OUT', NOW(), TRUE)
            """, (emp_id,))
    conn.commit()
    cursor.close()
    conn.close()

def get_attendance_logs():
    check_break_timeout()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT al.log_id, e.name, al.status, al.timestamp, al.is_break_exceeded
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        ORDER BY al.timestamp DESC
    """)
    logs = cursor.fetchall()
    # Convert to list of dicts for template compatibility
    logs = [{"log_id": log[0], "name": log[1], "status": log[2], "timestamp": log[3], "is_break_exceeded": log[4]} for log in logs]
    cursor.close()
    conn.close()
    return logs

def get_filtered_attendance_logs(year=None, month=None, day=None, employee_name=None):
    check_break_timeout()
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT al.log_id, e.name, al.status, al.timestamp, al.is_break_exceeded
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        WHERE 1=1
    """
    params = []
    if year:
        query += " AND YEAR(al.timestamp) = %s"
        params.append(year)
    if month:
        query += " AND MONTH(al.timestamp) = %s"
        params.append(month)
    if day:
        query += " AND DAY(al.timestamp) = %s"
        params.append(day)
    if employee_name:
        query += " AND e.name = %s"
        params.append(employee_name)
    query += " ORDER BY al.timestamp DESC"

    cursor.execute(query, params)
    logs = cursor.fetchall()
    # Convert to list of dicts
    logs = [{"log_id": log[0], "name": log[1], "status": log[2], "timestamp": log[3], "is_break_exceeded": log[4]} for log in logs]
    cursor.close()
    conn.close()
    return logs

def get_attendance_summary():
    logs = get_attendance_logs()
    if not logs:
        return pd.DataFrame()
    df = pd.DataFrame(logs)
    df.columns = ["log_id", "name", "status", "timestamp", "is_break_exceeded"]
    summary = df.groupby(["name", "status"]).size().reset_index(name="Count")
    return summary

def get_attendance_timeline(employee_name):
    logs = get_attendance_logs()
    if not logs:
        return pd.DataFrame()
    df = pd.DataFrame(logs)
    df = df[df["name"] == employee_name]
    return df

def get_todays_attendance_summary():
    check_break_timeout()
    today_date = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT e.name, al.status, al.timestamp
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        WHERE DATE(al.timestamp) = %s
        ORDER BY e.name, al.timestamp
    """
    cursor.execute(query, (today_date,))
    logs = cursor.fetchall()
    cursor.close()
    conn.close()

    summary = {}
    for name, status, timestamp in logs:
        if name not in summary:
            summary[name] = []
        summary[name].append({"status": status, "time": timestamp.strftime("%H:%M") if timestamp else "N/A"})
    return summary

def get_attendance_summary_by_date(date_str, employee_name=None):
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    check_break_timeout()
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT e.name, al.status, al.timestamp
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        WHERE DATE(al.timestamp) = %s
    """
    params = [date_str]
    if employee_name and employee_name != "all":
        query += " AND e.name = %s"
        params.append(employee_name)
    query += " ORDER BY e.name, al.timestamp"

    cursor.execute(query, params)
    logs = cursor.fetchall()
    cursor.close()
    conn.close()

    summary = {}
    for name, status, timestamp in logs:
        if name:
            if name not in summary:
                summary[name] = {"IN": [], "BREAK_START": [], "BREAK_END": [], "OUT": []}
            summary[name][status].append(timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "N/A")
    return summary

def get_filtered_attendance_metrics(year=None, month=None, day=None, employee_name=None):
    check_break_timeout()
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT e.name, al.status, al.timestamp, al.is_break_exceeded
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        WHERE 1=1
    """
    params = []
    if year:
        query += " AND YEAR(al.timestamp) = %s"
        params.append(year)
    if month:
        query += " AND MONTH(al.timestamp) = %s"
        params.append(month)
    if day:
        query += " AND DAY(al.timestamp) = %s"
        params.append(day)
    if employee_name:
        query += " AND e.name = %s"
        params.append(employee_name)
    query += " ORDER BY e.name, al.timestamp"

    cursor.execute(query, params)
    logs = cursor.fetchall()
    cursor.close()
    conn.close()

    metrics = {}
    for name, status, timestamp, is_break_exceeded in logs:
        if name:
            if name not in metrics:
                metrics[name] = {"IN": [], "OUT": [], "BREAK_START": [], "BREAK_END": [], "is_break_exceeded": False}
            metrics[name][status].append(timestamp)
            if is_break_exceeded:
                metrics[name]["is_break_exceeded"] = True

    df_data = []
    for name, times in metrics.items():
        in_times = times["IN"]
        out_times = times["OUT"]
        break_starts = times["BREAK_START"]
        break_ends = times["BREAK_END"]
        is_break_exceeded = times["is_break_exceeded"]

        total_hours = 0
        break_duration = 0
        if in_times and out_times:
            total_hours = (out_times[-1] - in_times[0]).total_seconds() / 3600
        for i in range(min(len(break_starts), len(break_ends))):
            break_duration += (break_ends[i] - break_starts[i]).total_seconds() / 3600
        if len(break_starts) > len(break_ends) and is_break_exceeded:
            break_duration += 2

        worked_hours = total_hours - break_duration if total_hours > 0 else 0
        df_data.append({
            "Employee Name": name,
            "Worked Hours": worked_hours,
            "Break Hours": break_duration,
            "Break Exceeded": "Yes" if is_break_exceeded else "No"
        })
    return pd.DataFrame(df_data)

def get_employee_logs_for_today(employee_name):
    today_date = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT al.log_id, al.status, al.timestamp
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        WHERE e.name = %s AND DATE(al.timestamp) = %s
        ORDER BY al.timestamp
    """
    cursor.execute(query, (employee_name, today_date))
    logs = cursor.fetchall()
    # Convert to list of dicts
    logs = [{"log_id": log[0], "status": log[1], "timestamp": log[2]} for log in logs]
    cursor.close()
    conn.close()
    return logs

@app.route('/video_feed')
def video_feed():
    try:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            logger.error("Could not open webcam at specified index.")
            return jsonify({"error": "Could not open webcam at specified index."}), 500
        logger.info("Webcam opened on index 0")

        def generate():
            global last_recognition
            try:
                while True:
                    success, frame = camera.read()
                    if not success or frame is None or frame.size == 0:
                        logger.error("Failed to read frame from webcam.")
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        face = gray[y:y+h, x:x+w]
                        face = preprocess_face(face)  # Ensure preprocessing matches training
                        try:
                            label, confidence = recognizer.predict(face)
                            logger.debug(f"Prediction - Label: {label}, Confidence: {confidence}, Label Dict: {label_dict}")
                            if confidence < 40 and label in label_dict:  # Lowered threshold to 40
                                name = label_dict.get(label, "Unknown")
                                current_time = datetime.now()
                                cooldown_remaining = 0

                                # Check cooldown
                                last_recog_time = last_recognition.get(name)
                                if last_recog_time:
                                    time_diff = (current_time - last_recog_time).total_seconds()
                                    if time_diff < COOLDOWN_SECONDS:
                                        cooldown_remaining = int(COLDOWN_SECONDS - time_diff)
                                    else:
                                        last_recognition[name] = current_time
                                else:
                                    last_recognition[name] = current_time

                                # Display name and cooldown
                                display_text = f"{name} (CD: {cooldown_remaining}s)"
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        except Exception as e:
                            logger.error(f"Error during prediction: {str(e)}")
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            cv2.putText(frame, "Error", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret:
                        logger.error("Failed to encode frame to JPEG.")
                        continue
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            finally:
                camera.release()
                logger.info("Webcam released.")

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video_feed: {str(e)}")
        return jsonify({"error": f"Failed to stream video feed: {str(e)}"}), 500

@app.route('/select_angle/<angle>', methods=['POST'])
def select_angle(angle):
    angle = angle.lower()
    if angle not in ["front", "right", "left"]:
        return jsonify({"status": "error", "message": "Invalid angle specified."})
    
    session["selected_angle"] = angle
    session.modified = True
    return jsonify({"status": "success", "message": f"Selected {angle} angle. Please position your face and capture."})

@app.route('/capture_single_image/<angle>', methods=['POST'])
def capture_single_image(angle):
    angle = angle.lower()
    if angle not in ["front", "right", "left"]:
        return jsonify({"status": "error", "message": "Invalid angle specified."})

    employee_name = session.get("employee_name")
    if not employee_name:
        return jsonify({"status": "error", "message": "Employee name not set."})

    face_dir = os.path.join(FACE_DATA_PATH, employee_name)
    os.makedirs(face_dir, exist_ok=True)

    existing_images = [f for f in os.listdir(face_dir) if f.startswith("img") and f.endswith('.jpg')]
    start_num = len(existing_images) + 1

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        return jsonify({"status": "error", "message": "Unable to access webcam."})

    try:
        success, frame = camera.read()
        if not success or frame is None:
            return jsonify({"status": "error", "message": "Failed to read frame from webcam."})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected in the frame. Please position your face correctly."})

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = preprocess_face(face)  # Preprocess before saving
            image_path = os.path.join(face_dir, f"img{start_num}.jpg")
            cv2.imwrite(image_path, face)
            
            session["captured_counts"][angle] = session["captured_counts"].get(angle, 0) + 1
            updated_images = [f for f in os.listdir(face_dir) if f.startswith("img") and f.endswith('.jpg')]
            session["total_captured"] = len(updated_images)
            logger.info(f"After capturing single image for {angle}: total_captured = {session['total_captured']}, captured_counts = {session['captured_counts']}")
            session.modified = True
            break

        return jsonify({"status": "success", "message": f"Captured image for {angle} angle."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during capture: {str(e)}"})
    finally:
        camera.release()

@app.route('/capture_image', methods=['POST'])
def capture_image():
    employee_name = session.get("employee_name")
    if not employee_name:
        return jsonify({"status": "error", "message": "Employee name not set."})

    angle = session.get("selected_angle", "front")  # Default to front if no angle selected
    face_dir = os.path.join(FACE_DATA_PATH, employee_name)
    os.makedirs(face_dir, exist_ok=True)

    existing_images = [f for f in os.listdir(face_dir) if f.startswith("img") and f.endswith('.jpg')]
    start_num = len(existing_images) + 1

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        return jsonify({"status": "error", "message": "Unable to access webcam."})

    try:
        success, frame = camera.read()
        if not success or frame is None:
            return jsonify({"status": "error", "message": "Failed to read frame from webcam."})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected in the frame. Please position your face correctly."})

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = preprocess_face(face)  # Preprocess before saving
            image_path = os.path.join(face_dir, f"img{start_num}.jpg")
            cv2.imwrite(image_path, face)
            
            session["captured_counts"][angle] = session["captured_counts"].get(angle, 0) + 1
            updated_images = [f for f in os.listdir(face_dir) if f.startswith("img") and f.endswith('.jpg')]
            session["total_captured"] = len(updated_images)
            logger.info(f"After capturing image: total_captured = {session['total_captured']}, captured_counts = {session['captured_counts']}")
            session.modified = True
            break

        return jsonify({"status": "success", "message": f"Captured image for {angle} angle."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during capture: {str(e)}"})
    finally:
        camera.release()

@app.route('/capture_images_automatically/<angle>', methods=['POST'])
def capture_images_automatically(angle):
    angle = angle.lower()
    if angle not in ["front", "right", "left"]:
        return jsonify({"status": "error", "message": "Invalid angle specified."})

    employee_name = session.get("employee_name")
    if not employee_name:
        return jsonify({"status": "error", "message": "Employee name not set."})

    session["capturing"][angle] = True
    session["captured_counts"][angle] = session["captured_counts"].get(angle, 0)
    session.modified = True

    face_dir = os.path.join(FACE_DATA_PATH, employee_name)
    os.makedirs(face_dir, exist_ok=True)

    existing_images = [f for f in os.listdir(face_dir) if f.startswith("img") and f.endswith('.jpg')]
    start_num = len(existing_images) + 1

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        session["capturing"][angle] = False
        session.modified = True
        return jsonify({"status": "error", "message": "Unable to access webcam."})

    try:
        captured = 0
        max_images = 10  # Capture 10 images per angle (3 angles = 30 images)
        while captured < max_images:
            success, frame = camera.read()
            if not success or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face = preprocess_face(face)  # Preprocess before saving
                    image_path = os.path.join(face_dir, f"img{start_num + captured}.jpg")
                    cv2.imwrite(image_path, face)
                    captured += 1
                    session["captured_counts"][angle] = session["captured_counts"].get(angle, 0) + 1
                    updated_images = [f for f in os.listdir(face_dir) if f.startswith("img") and f.endswith('.jpg')]
                    session["total_captured"] = len(updated_images)
                    logger.info(f"After capturing image {captured} for {angle}: total_captured = {session['total_captured']}, captured_counts = {session['capturing']}")
                    session.modified = True
                    break

            time.sleep(1)

        session["capturing"][angle] = False
        session.modified = True
        return jsonify({"status": "success", "message": f"Captured {captured} images for {angle} angle."})
    except Exception as e:
        session["capturing"][angle] = False
        session.modified = True
        return jsonify({"status": "error", "message": f"Error during capture: {str(e)}"})
    finally:
        camera.release()

@app.route('/get_capture_status')
def get_capture_status():
    employee_name = session.get("employee_name")
    captured_images = []
    if employee_name:
        face_dir = os.path.join(FACE_DATA_PATH, employee_name)
        if os.path.exists(face_dir):
            captured_images = [f for f in os.listdir(face_dir) if f.startswith("img") and f.endswith('.jpg')]
            session["total_captured"] = len(captured_images)
            logger.info(f"In get_capture_status: total_captured = {session['total_captured']}, captured_counts = {session['captured_counts']}")
            session.modified = True

    return jsonify({
        "capturing": session.get("capturing", {"front": False, "right": False, "left": False}),
        "captured_counts": session.get("captured_counts", {"front": 0, "right": 0, "left": 0}),
        "total_captured": session.get("total_captured", 0),
        "selected_angle": session.get("selected_angle", None),
        "captured_images": captured_images
    })

@app.route('/')
def home():
    initialize_session()
    return render_template('home.html')

@app.route('/add_employee', methods=['GET', 'POST'])
def add_employee():
    initialize_session()
    employees = get_employees()
    message = None
    error = None
    captured_images = []
    can_train = False

    if request.method == 'POST':
        logger.info(f"Received POST data: {request.form}")
        if 'new_employee' in request.form:
            new_employee = request.form.get('new_employee', '').strip()
            new_employee = re.sub(r'[^a-zA-Z0-9\s]', '', new_employee)
            if not new_employee:
                error = "Employee name cannot be empty."
            elif new_employee in employees:
                error = f"Employee '{new_employee}' already exists. Please choose a different name."
            else:
                session["employee_name"] = new_employee
                session["captured_counts"] = {"front": 0, "right": 0, "left": 0}
                session["capturing"] = {"front": False, "right": False, "left": False}
                session["total_captured"] = 0
                session["selected_angle"] = None
                session.modified = True

        elif 'self_train' in request.form or request.form.get('action') == 'train':
            employee_name = session.get("employee_name")
            total_captured = session.get("total_captured", 0)
            logger.info(f"Self-train requested for {employee_name} with {total_captured} images")
            response = {"message": None, "error": None}

            if not employee_name:
                response["error"] = "Employee name not set."
            elif total_captured < 30:
                response["error"] = f"Please capture at least 30 images before training. Currently captured: {total_captured}."
            else:
                try:
                    # Add employee to database
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO employees (name, folder_name) VALUES (%s, %s)", (employee_name, employee_name))
                    conn.commit()
                    cursor.close()
                    conn.close()
                    logger.info(f"Added {employee_name} to database")

                    # Train the model
                    success, msg = train_model(employee_name)
                    if success:
                        response["message"] = msg
                        session["training_message"] = response["message"]
                        session["training_error"] = None
                    else:
                        response["error"] = msg
                        session["training_error"] = msg
                        session["training_message"] = None
                        logger.error(f"Training failed: {msg}")
                except Exception as e:
                    logger.error(f"Exception during self-train: {str(e)}")
                    response["error"] = f"Error during training: {str(e)}"
                    session["training_error"] = response["error"]

                # Clear session for next employee
                session["employee_name"] = None
                session["captured_counts"] = {"front": 0, "right": 0, "left": 0}
                session["capturing"] = {"front": False, "right": False, "left": False}
                session["total_captured"] = 0
                session["selected_angle"] = None
                session.modified = True
                captured_images = []

            return jsonify(response)

    employee_name = session.get("employee_name")
    if employee_name:
        face_dir = os.path.join(FACE_DATA_PATH, employee_name)
        if os.path.exists(face_dir):
            captured_images = [f for f in os.listdir(face_dir) if f.startswith("img") and f.endswith('.jpg')]
            total_images = len(captured_images)
            session["total_captured"] = total_images
            can_train = total_images >= 30
            logger.info(f"In add_employee: employee_name = {employee_name}, total_captured = {total_images}, captured_images = {len(captured_images)}")
            counts = session["captured_counts"]
            total_counts = sum(counts.values())
            if total_counts != total_images:
                if total_counts > 0:
                    scale = total_images / total_counts
                    counts = {angle: min(int(count * scale), 10) for angle, count in counts.items()}
                    total_assigned = sum(counts.values())
                    if total_assigned < total_images:
                        remaining = total_images - total_assigned
                        for angle in ["front", "right", "left"]:
                            if remaining == 0:
                                break
                            if counts[angle] < 10:
                                increment = min(10 - counts[angle], remaining)
                                counts[angle] += increment
                                remaining -= increment
                else:
                    counts = {"front": min(total_images, 10), "right": 0, "left": 0}
                    if total_images > 10:
                        counts["right"] = min(total_images - 10, 10)
                    if total_images > 20:
                        counts["left"] = min(total_images - 20, 10)
                session["captured_counts"] = counts
            session.modified = True
            logger.info(f"In add_employee: updated captured_counts = {session['captured_counts']}")

    return render_template('add_employee.html',
                          employees=employees,
                          employee_name=employee_name,
                          captured_counts=session.get("captured_counts", {"front": 0, "right": 0, "left": 0}),
                          capturing=session.get("capturing", {"front": False, "right": False, "left": False}),
                          total_captured=session.get("total_captured", 0),
                          captured_images=captured_images if employee_name and not request.form.get('self_train') else [],
                          can_train=can_train,
                          message=message or session.pop("training_message", None),
                          error=error or session.pop("training_error", None))

@app.route('/log_attendance', methods=['GET', 'POST'])
def log_attendance():
    initialize_session()
    employees = get_employees()
    todays_summary = get_todays_attendance_summary()
    message = None
    error = None
    selected_employee = None
    employee_logs = []

    if request.method == 'POST':
        if 'recognize' in request.form:
            status, message, error = recognize_and_log(recognizer, label_dict)
            if status in ["recognized", "stopped"]:
                todays_summary = get_todays_attendance_summary()
            else:
                error = message
                message = None

        elif 'select_employee' in request.form:
            selected_employee = request.form.get('employee_name')
            if selected_employee:
                employee_logs = get_employee_logs_for_today(selected_employee)
            else:
                error = "Please select an employee."

        elif 'update_logs' in request.form:
            selected_employee = request.form.get('employee_name')
            updated_logs = []
            for key, value in request.form.items():
                if key.startswith('log_id_'):
                    log_id = int(value)
                    status = request.form.get(f'status_{log_id}')
                    timestamp_str = request.form.get(f'timestamp_{log_id}')
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M')
                        updated_logs.append((log_id, status, timestamp))
                    except ValueError:
                        error = f"Invalid timestamp format for Log ID {log_id}. Use YYYY-MM-DDThh:mm (e.g., 2025-06-04T14:30)."
                        break

            if not error and updated_logs:
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    for log_id, status, timestamp in updated_logs:
                        cursor.execute("""
                            UPDATE attendance_log 
                            SET status = %s, timestamp = %s 
                            WHERE log_id = %s
                        """, (status, timestamp, log_id))
                    conn.commit()
                    message = "Attendance logs updated successfully!"
                    employee_logs = get_employee_logs_for_today(selected_employee)
                    todays_summary = get_todays_attendance_summary()
                except Exception as e:
                    conn.rollback()
                    error = f"Error updating logs: {str(e)}"
                finally:
                    cursor.close()
                    conn.close()

        elif 'add_break' in request.form:
            selected_employee = request.form.get('employee_name')
            break_start = request.form.get('new_break_start')
            break_end = request.form.get('new_break_end')

            if not selected_employee:
                error = "Please select an employee."
            else:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT emp_id FROM employees WHERE name = %s", (selected_employee,))
                result = cursor.fetchone()
                if not result:
                    error = f"Employee {selected_employee} not found."
                else:
                    emp_id = result[0]
                    today_date = datetime.now().strftime("%Y-%m-%d")
                    updates = []
                    for status, time in [('BREAK_START', break_start), ('BREAK_END', break_end)]:
                        if time:
                            try:
                                new_timestamp = datetime.strptime(f"{today_date} {time}", '%Y-%m-%d %H:%M')
                                cursor.execute("""
                                    INSERT INTO attendance_log (emp_id, status, timestamp)
                                    VALUES (%s, %s, %s)
                                """, (emp_id, status, new_timestamp))
                                updates.append(f"Added {status} at {time}")
                            except ValueError:
                                error = f"Invalid time format for {status}. Use HH:MM (e.g., 14:30)."
                                break

                    if updates and not error:
                        try:
                            conn.commit()
                            message = f"Added break for {selected_employee}: {', '.join(updates)}"
                            employee_logs = get_employee_logs_for_today(selected_employee)
                            todays_summary = get_todays_attendance_summary()
                        except Exception as e:
                            conn.rollback()
                            error = f"Error adding break: {str(e)}"
                    elif not updates and not error:
                        error = "Please provide at least one time to add a break."
                cursor.close()
                conn.close()

    return render_template('log_attendance.html',
                          employees=employees,
                          todays_summary=todays_summary,
                          selected_employee=selected_employee,
                          employee_logs=employee_logs,
                          message=message,
                          error=error)

@app.route('/admin_panel', methods=['GET', 'POST'])
def admin_panel():
    initialize_session()
    employees = get_employees()
    logs = get_attendance_logs()
    logs_df = pd.DataFrame(logs) if logs else pd.DataFrame()
    summary_df = get_attendance_summary()

    message = None
    error = None
    authorized = False
    selected_year = None
    selected_month = None
    selected_day = None
    selected_employee = None
    selected_date = datetime.now().strftime("%Y-%m-%d")

    if request.method == 'POST':
        if 'password' in request.form:
            if request.form['password'] == "admin123":
                session['admin_panel_authorized'] = True
                authorized = True
            else:
                error = "Incorrect password. Please try again."
        else:
            authorized = session.get('admin_panel_authorized', False)

        if authorized:
            if 'delete_employee_btn' in request.form:
                employee_name = request.form.get('delete_employee')
                if not employee_name:
                    error = "Please select an employee to delete."
                else:
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("SELECT emp_id FROM employees WHERE name = %s", (employee_name,))
                        result = cursor.fetchone()
                        if not result:
                            error = f"Employee {employee_name} not found."
                        else:
                            emp_id = result[0]
                            cursor.execute("DELETE FROM attendance_log WHERE emp_id = %s", (emp_id,))
                            cursor.execute("DELETE FROM employees WHERE emp_id = %s", (emp_id,))
                            conn.commit()
                            face_data_dir = os.path.join(FACE_DATA_PATH, employee_name)
                            if os.path.exists(face_data_dir):
                                shutil.rmtree(face_data_dir)
                            success, msg = train_model()
                            if not success:
                                error = f"Employee {employee_name} deleted, but failed to retrain model: {msg}"
                            else:
                                message = f"Employee {employee_name} and their data have been deleted successfully. Model retrained."
                        cursor.close()
                        conn.close()
                    except Exception as e:
                        error = f"Error deleting employee: {str(e)}"
                    employees = get_employees()
                    logs = get_attendance_logs()
                    logs_df = pd.DataFrame(logs) if logs else pd.DataFrame()
                    summary_df = get_attendance_summary()

            elif 'filter_logs' in request.form:
                selected_year = request.form.get('year')
                selected_month = request.form.get('month')
                selected_date = request.form.get('selected_date')
                selected_employee = request.form.get('selected_employee')

                if selected_date:
                    try:
                        date_obj = datetime.strptime(selected_date, "%Y-%m-%d")
                        selected_year = date_obj.year
                        selected_month = date_obj.month
                        selected_day = date_obj.day
                    except ValueError:
                        error = "Invalid date format. Please use YYYY-MM-DD."
                else:
                    today = datetime.now()
                    selected_year = today.year
                    selected_month = today.month
                    selected_day = today.day
                    selected_date = today.strftime("%Y-%m-%d")

                if selected_year and selected_year != "":
                    selected_year = int(selected_year)
                else:
                    selected_year = None
                if selected_month and selected_month != "":
                    selected_month = int(selected_month)
                else:
                    selected_month = None
                if selected_day and selected_day != "":
                    selected_day = int(selected_day)
                else:
                    selected_day = None
                if selected_employee == "all":
                    selected_employee = None

                logs = get_filtered_attendance_logs(selected_year, selected_month, selected_day, selected_employee)
                logs_df = pd.DataFrame(logs) if logs else pd.DataFrame()
                summary_df = get_attendance_summary()

            elif 'update_logs' in request.form:
                updated_logs = []
                for key, value in request.form.items():
                    if key.startswith('log_id_'):
                        log_id = int(value)
                        status = request.form.get(f'status_{log_id}')
                        timestamp_str = request.form.get(f'timestamp_{log_id}')
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M')
                            updated_logs.append((log_id, status, timestamp))
                        except ValueError:
                            error = f"Invalid timestamp format for Log ID {log_id}. Use YYYY-MM-DDThh:mm (e.g., 2025-06-02T14:30)."
                            break

                if not error and updated_logs:
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        for log_id, status, timestamp in updated_logs:
                            cursor.execute("""
                                UPDATE attendance_log 
                                SET status = %s, timestamp = %s 
                                WHERE log_id = %s
                            """, (status, timestamp, log_id))
                        conn.commit()
                        message = "Attendance logs updated successfully!"
                        logs = get_attendance_logs()
                        logs_df = pd.DataFrame(logs) if logs else pd.DataFrame()
                        summary_df = get_attendance_summary()
                    except Exception as e:
                        conn.rollback()
                        error = f"Error updating logs: {str(e)}"
                    finally:
                        cursor.close()
                        conn.close()

    date_summary = get_attendance_summary_by_date(selected_date, selected_employee)

    # Plotly Visualizations
    metrics_df = get_filtered_attendance_metrics(selected_year, selected_month, selected_day, selected_employee)
    metrics_fig = None
    if not metrics_df.empty:
        fig = px.bar(metrics_df, x="Employee Name", y=["Worked Hours", "Break Hours"],
                     title="Attendance Metrics",
                     labels={"value": "Hours", "variable": "Metric"},
                     barmode="group")
        metrics_fig = fig.to_html(full_html=False, include_plotlyjs='cdn')

    summary_fig = None
    if not summary_df.empty:
        fig = px.bar(summary_df, x="name", y="Count", color="status",
                     title="Attendance Events per Employee",
                     labels={"Count": "Number of Events"},
                     barmode="group")
        summary_fig = fig.to_html(full_html=False, include_plotlyjs='cdn')

    timeline_fig = None
    default_employee = employees[0] if employees else None
    timeline_employee = request.form.get('timeline_employee', default_employee)
    if timeline_employee:
        timeline_df = get_attendance_timeline(timeline_employee)
        if not timeline_df.empty:
            fig = px.line(timeline_df, x="timestamp", y="status",
                         title=f"Attendance Timeline for {timeline_employee}",
                         markers=True)
            timeline_fig = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Chart.js Visualization
    summary_data = {}
    for log in logs:
        emp_name, status = log["name"], log["status"]
        if emp_name not in summary_data:
            summary_data[emp_name] = {'IN': 0, 'BREAK_START': 0, 'BREAK_END': 0, 'OUT': 0}
        summary_data[emp_name][status] += 1

    summary_labels = list(summary_data.keys())
    summary_datasets = [
        {
            'label': 'IN',
            'data': [summary_data[emp]['IN'] for emp in summary_labels],
            'backgroundColor': '#2ecc71'
        },
        {
            'label': 'BREAK_START',
            'data': [summary_data[emp]['BREAK_START'] for emp in summary_labels],
            'backgroundColor': '#e67e22'
        },
        {
            'label': 'BREAK_END',
            'data': [summary_data[emp]['BREAK_END'] for emp in summary_labels],
            'backgroundColor': '#3498db'
        },
        {
            'label': 'OUT',
            'data': [summary_data[emp]['OUT'] for emp in summary_labels],
            'backgroundColor': '#e74c3c'
        }
    ]

    summary_chart = {
        "type": "bar",
        "data": {
            "labels": summary_labels,
            "datasets": summary_datasets
        },
        "options": {
            "plugins": {
                "title": {
                    "display": True,
                    "text": "Attendance Events per Employee (Chart.js)"
                }
            },
            "scales": {
                "x": {"stacked": False},
                "y": {"stacked": False, "beginAtZero": True, "title": {"display": True, "text": "Number of Events"}}
            }
        }
    }

    events = []
    if logs:
        for log in logs:
            employee_name, status, timestamp = log["name"], log["status"], log["timestamp"]
            if timestamp:
                events.append({
                    "title": f"{employee_name}: {status}",
                    "start": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "end": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "color": {"IN": "green", "OUT": "red", "BREAK_START": "orange", "BREAK_END": "blue"}.get(status, "gray")
                })

    years = list(range(2020, 2026))
    months = list(range(1, 13))

    return render_template('admin_panel.html',
                          authorized=authorized,
                          employees=employees,
                          logs=logs,
                          date_summary=date_summary,
                          summary_fig=summary_fig,
                          summary_chart=summary_chart,
                          timeline_fig=timeline_fig,
                          metrics_fig=metrics_fig,
                          selected_employee=selected_employee,
                          timeline_employee=timeline_employee,
                          events=events,
                          years=years,
                          months=months,
                          selected_year=selected_year,
                          selected_month=selected_month,
                          selected_date=selected_date,
                          message=message,
                          error=error)

@app.route('/download_logs')
def download_logs():
    logs = get_attendance_logs()
    df = pd.DataFrame(logs)
    df.columns = ["Log ID", "Employee Name", "Status", "Timestamp", "Break Exceeded"]
    csv = df.to_csv(index=False)
    return send_file(
        BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='attendance_logs.csv'
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)