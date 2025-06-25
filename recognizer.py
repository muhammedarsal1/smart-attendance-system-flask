import cv2
import numpy as np
from datetime import datetime, timedelta
from db_config import get_db_connection
import time

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
if face_cascade.empty():
    raise RuntimeError("Error: Could not load face cascade classifier.")

# Dictionary to track the last recognition time for each employee
last_recognition_times = {}
COOLDOWN_SECONDS = 300  # 5 minutes cooldown


def get_last_status_today(emp_id, today_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT status, timestamp
        FROM attendance_log
        WHERE emp_id = %s
        AND DATE(timestamp) = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """
    cursor.execute(query, (emp_id, today_date))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result if result else (None, None)


def determine_next_status(last_status):
    if last_status is None:
        return "IN"
    elif last_status == "IN":
        return "BREAK_START"
    elif last_status == "BREAK_START":
        return "BREAK_END"
    elif last_status == "BREAK_END":
        return "BREAK_START"  # Allow multiple breaks
    elif last_status == "OUT":
        return None  # Cannot log further events after OUT
    return None


def log_attendance(emp_id, status, break_exceeded=False):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        if break_exceeded:
            cursor.execute("""
                INSERT INTO attendance_log (emp_id, status, timestamp, is_break_exceeded)
                VALUES (%s, %s, NOW(), TRUE)
            """, (emp_id, status))
        else:
            cursor.execute("""
                INSERT INTO attendance_log (emp_id, status, timestamp)
                VALUES (%s, %s, NOW())
            """, (emp_id, status))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Successfully logged attendance for emp_id={emp_id} with status={status}")
        return True
    except Exception as e:
        print(f"Error logging attendance: {e}")
        return False


def preprocess_face(face):
    face = cv2.resize(face, (100, 100))
    face = cv2.equalizeHist(face)
    face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)
    return face


def recognize_and_log(recognizer, label_dict):
    cap = None
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Webcam opened on index {i}")
            break
    else:
        return "error", "Failed to open webcam.", None

    try:
        print("Starting real-time face recognition. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = preprocess_face(face)

                label, confidence = recognizer.predict(face)
                if label in label_dict and confidence < 80:
                    employee_name = label_dict[label]
                    current_time = datetime.now()

                    if employee_name in last_recognition_times:
                        last_time = last_recognition_times[employee_name]
                        if (current_time - last_time).total_seconds() < COOLDOWN_SECONDS:
                            remaining = int(COOLDOWN_SECONDS - (current_time - last_time).total_seconds())
                            cv2.putText(frame, f"{employee_name} (CD: {remaining}s)", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            continue

                    last_recognition_times[employee_name] = current_time

                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT emp_id FROM employees WHERE name = %s", (employee_name,))
                    result = cursor.fetchone()

                    if result:
                        emp_id = result[0]
                        today_date = datetime.now().strftime("%Y-%m-%d")
                        last_status, last_timestamp = get_last_status_today(emp_id, today_date)

                        if last_status == "BREAK_START" and last_timestamp:
                            time_diff = (datetime.now() - last_timestamp).total_seconds()
                            if time_diff >= 9000:  # 2.5 hours
                                if log_attendance(emp_id, "OUT", break_exceeded=True):
                                    cv2.putText(frame, f"{employee_name} - Auto OUT", (x, y-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                continue

                        next_status = determine_next_status(last_status)
                        if next_status and log_attendance(emp_id, next_status):
                            cv2.putText(frame, f"{employee_name} - {next_status}", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f"{employee_name} - OUT", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Unknown Employee", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return "stopped", "Real-time recognition stopped by user.", None

    finally:
        cap.release()
        cv2.destroyAllWindows()


def manual_log(employee_name, status, label_dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT emp_id FROM employees WHERE name = %s", (employee_name,))
    result = cursor.fetchone()
    if not result:
        cursor.close()
        conn.close()
        return False, f"Employee {employee_name} not found."

    emp_id = result[0]
    today_date = datetime.now().strftime("%Y-%m-%d")
    last_status, _ = get_last_status_today(emp_id, today_date)
    expected_status = determine_next_status(last_status)

    if expected_status != status:
        cursor.close()
        conn.close()
        return False, f"Cannot log {status} for {employee_name}. Expected status: {expected_status or 'none'}."

    if log_attendance(emp_id, status):
        last_recognition_times[employee_name] = datetime.now()
        cursor.close()
        conn.close()
        return True, f"Manually logged {status} for {employee_name}."
    else:
        cursor.close()
        conn.close()
        return False, "Failed to log attendance."


if __name__ == "__main__":
    print("This module is meant to be imported and used with app.py.")