# Smart Attendance System Using Face Recognition

A modern, **AI-powered application** designed to automate attendance tracking for employees or students using **real-time face recognition**. This system replaces traditional manual or fingerprint-based methods, offering a **fast, accurate, and contactless** solution for attendance management.

---

## 🚀 Features

- **Real-Time Face Recognition**: Automatically detects and recognizes faces to log attendance seamlessly.
- **Event Tracking**: Captures **IN**, **BREAK START**, **BREAK END**, and **OUT** events for comprehensive attendance monitoring.
- **Duplicate Prevention**: Implements a 5-minute cooldown to avoid duplicate entries.
- **Auto-Logout**: Automatically logs an **OUT** event if a break exceeds 2.5 hours.
- **Admin Panel**: Intuitive interface for managing employees and attendance logs.
- **Report Generation**: Export attendance data as **CSV** for easy reporting.
- **Interactive Visualizations**: View attendance trends through dynamic timelines and charts.

---

## 📂 Project Structure

```bash
Smart-Attendance-System/
├── app.py                # Flask backend for API and server logic
├── db_config.py          # Database configuration and connection setup
├── recognizer.py         # Face detection and attendance logging logic
├── trainer.yml           # Trained LBPH model for face recognition
├── label_dict.npz       # Mapping of face labels to identifiers
├── templates/            # HTML templates for the frontend
└── static/               # Static assets (CSS, JavaScript, images)
```

---

## 🧠 Face Recognition Workflow

1. **Face Detection**: Utilizes the HaarCascade Classifier for accurate face detection.
2. **Face Recognition**: Employs OpenCV's LBPH (Local Binary Patterns Histograms) algorithm for reliable identification.
3. **Attendance Logging**: Records events in a MySQL database with timestamps.
4. **Duplicate Prevention**: Enforces a 5-minute cooldown to prevent redundant logs.
5. **Break Monitoring**: Automatically triggers an **OUT** event if a break exceeds 2.5 hours.

---

## 📊 Attendance Flow

1. **Initial Log**: Registers an **IN** event for the first scan of the day.
2. **Break Management**: Logs **BREAK START** and **BREAK END** for subsequent scans.
3. **Final Log**: Records an **OUT** event to close the attendance cycle.
4. **Auto-Logout**: Automatically logs **OUT** if a break exceeds 2.5 hours.

---

## 🛠️ Technologies Used

- **Backend**: Python, Flask
- **Database**: MySQL
- **Face Recognition**: OpenCV, NumPy
- **Data Visualization**: Pandas, Plotly, Chart.js
- **Frontend**: HTML, CSS, JavaScript

---

## 📊 Admin Panel Features

- **Employee Management**: Add, update, or remove employee profiles.
- **Log Management**: View, edit, and filter attendance logs by date or employee.
- **Data Visualization**: Analyze attendance trends with interactive charts and timelines.
- **Export Functionality**: Generate and download attendance reports in CSV format.

---

## ✅ Conclusion

The **Smart Attendance System** leverages AI-powered face recognition to provide a **secure, efficient, and contactless** solution for attendance tracking. Ideal for schools, colleges, offices, and organizations, it streamlines attendance management while offering robust reporting and visualization capabilities.

---

## 👨‍💻 Developed By

- **Sabith**
- **Arsal**

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.