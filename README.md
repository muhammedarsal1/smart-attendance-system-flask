#👨‍💼 Smart Attendance System 🚀

A **Flask-based web application** that automates employee attendance using **real-time face recognition**, **image capture**, **MySQL database**, and **interactive dashboards**.

---

## 📸 Features

✅ **Face Detection & Recognition**  
✅ **Add Employees & Train Model**  
✅ **Real-Time Attendance Logging (IN, OUT, BREAK)**  
✅ **Manual Log Updates**  
✅ **Admin Panel** with Visualizations  
✅ **Downloadable Attendance Logs**  
✅ **Break Time Monitoring & Auto OUT**  
✅ **Angle-wise Face Capture (Front, Left, Right)**  
✅ **Beautiful UI with Clocks and Animations**  

---

## 🖼️ Screenshots

> Add screenshots or demo GIFs here  
> `home.html`, `add_employee.html`, `log_attendance.html`, `admin_panel.html`

---

## 🛠️ Tech Stack

| Layer           | Technology                        |
|----------------|------------------------------------|
| Backend         | Flask (Python)                    |
| Database        | MySQL                             |
| Face Detection  | OpenCV (Haarcascade + LBPH)       |
| Frontend        | HTML5, CSS3, Bootstrap, JS        |
| Visualizations  | Plotly, Chart.js                  |
| Data Export     | Pandas CSV                        |

---

## 📂 Project Structure

Smart-Attendance-System/
│
├── app.py # Main Flask application
├── recognizer.py # Face recognition & attendance logic
├── trainer.py # LBPH face trainer
├── save_faces.py # Image capture logic
├── db_config.py # DB connection & init
├── templates/ # HTML pages
│ ├── base.html
│ ├── home.html
│ ├── add_employee.html
│ ├── log_attendance.html
│ ├── admin_panel.html
│ └── error.html
├── static/ # CSS, JS, images
├── face_data/ # Stored images of employees
├── trainer.yml # Trained model
├── labels.txt # Label-name mapping
└── .env # Environment variables (DB creds)


---

## ⚙️ Setup Instructions

### 🔧 Prerequisites

- Python 3.8+
- MySQL Server installed
- `pip install` packages:  
  `flask opencv-python numpy pandas plotly python-dotenv mysql-connector-python`

---

### 🔁 Installation Steps


git clone https://github.com/your-username/smart-attendance-system.git
cd smart-attendance-system

# Create .env file
`cp .env.example .env  # or manually create it`

# Install required packages
`pip install -r requirements.txt`

# Initialize database and tables
`python -c "from db_config import init_db; init_db()"`

# Start the application
`python app.py`

###🧾 .env Format
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=attendance_db

###💡 How It Works

1- Admin adds a new employee.

2- Capture 30 face images from 3 angles.

3- Train the model with Self-Train Model button.

4- Recognize and auto-log attendance using webcam.

5- Use Admin Panel to:

  - View, edit, delete logs

  - View summaries

  - Plot graphs

  - Download CSV logs


###📊 Admin Panel Highlights
  🔐 Password-Protected Access

  🧍‍♂️ Employee Deletion

  🗂️ Log Filtering (by date, employee)

  📝 Log Editing (Status, Timestamp)

  📈 Visualizations (Plotly, Chart.js)

  📤 Download Logs as CSV

###📤 Export Attendance Logs
  Admin Panel ➝ "Download Attendance Logs as CSV"
  Generates attendance_logs.csv with full records.

###📅 Auto OUT Logic
If BREAK_START exceeds 2.5 hours (9000 seconds), the system automatically logs OUT with a break_exceeded flag.

###🎨 UI Features
Animated Gradient Backgrounds

Live Analog + Digital Clock (IST)

Typing Title Animation

Responsive Cards for Navigation

Stylish Forms, Banners, Tables

###✅ Authors
👨‍💻 Arsal —— Developer 
👨‍💻 Sabith —— Co-Developer 


###📌 Future Improvements
  Email/SMS notifications

  Dashboard calendar view

  FaceNet or DNN-based recognition

  QR code or RFID attendance

📄 License
  This project is licensed under the MIT License – free to use, modify, and distribute.




