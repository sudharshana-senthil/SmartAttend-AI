# 🎓 SmartAttend AI System
### Production-Ready Zero-Proxy Biometric, Spatial & Period-Wise Attendance Verification Platform

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-orange.svg)](https://developers.google.com/mediapipe)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-v3.4-38B2AC.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

**SmartAttend AI** is a zero-proxy, period-wise attendance management system designed for colleges and universities. Traditional attendance systems (paper roll calls, static barcode cards, or generic QR codes) are vulnerable to buddy punching, location spoofing, and photo spoofing.

SmartAttend resolves these issues through **"The Verification Triangle"**:
1. 📍 **Spatial Geofence Verification**: Compares instructor device GPS with student mobile GPS using the spherical Haversine formula (enforces a strict 30m classroom boundary).
2. 😊 **3D Vitality Liveness Verification**: Integrates client-side MediaPipe 3D FaceMesh landmark tracking to confirm that a live human is physically present, blocking printed photos and pre-recorded videos.
3. 🧬 **Biometric Identity Matching**: Extracts a 128-dimensional facial feature embedding vector using OpenCV and matches it against the enrolled baseline using Cosine Similarity ($\ge 70\%$).

All of this works **directly inside mobile and desktop web browsers**—no proprietary biometric hardware or native app store downloads required!

---

## ✨ Key Features

- **Period-Wise Attendance**: Explicit selection for academic periods (**Period 1 to Period 7**) with associated course name and faculty in-charge.
- **Dynamic LAN QR Code**: Automatically detects host machine's local Wi-Fi IP (e.g. `192.168.x.x`) so mobile devices on the same network connect instantly upon scanning.
- **Live Verification Roster**: Real-time auto-polling updates the instructor's dashboard table every 3 seconds as students verify.
- **Master Student Directory**: View cumulative attendance statistics across all registered students in the institution.
- **One-Click CSV Export**: Download comprehensive period-specific and master institution reports containing:
  - Student ID, Name, Department
  - Staff / Faculty Name
  - Course Name & Period Name
  - Verification Timestamps
  - Biometric Similarity & Distance Metrics
- **Responsive Dark-Mode Interface**: Built with Tailwind CSS and FontAwesome for a clean, modern experience on all screen sizes.

---

## 🏗️ Architecture & Verification Triangle

```
                            +-----------------------+
                            | Faculty Starts Session|
                            | (GPS Locked, Period)  |
                            +-----------+-----------+
                                        |
                                        v
                            +-----------------------+
                            | Dynamic QR Code Link  |
                            +-----------+-----------+
                                        |
                                        v
                            +-----------------------+
                            |  Student Scans QR &   |
                            |   Opens /student      |
                            +-----------+-----------+
                                        |
        +-------------------------------+-------------------------------+
        |                               |                               |
        v                               v                               v
[1. Spatial Check]             [2. Vitality Check]             [3. Identity Check]
Haversine Distance <= 30m?     MediaPipe 3D Liveness Passed?   Cosine Similarity >= 70%?
        |                               |                               |
        +-------------------------------+-------------------------------+
                                        |
                                        v
                        +-------------------------------+
                        | Attendance Verified (PRESENT) |
                        | Logged to SQLite & Live Table |
                        +-------------------------------+
```

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
- **Computer Vision & Math**: [OpenCV](https://opencv.org/) (`cv2`), [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/)
- **Liveness Detection**: [Google MediaPipe FaceMesh](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- **Database & ORM**: [SQLAlchemy](https://www.sqlalchemy.org/) with [SQLite 3](https://sqlite.org/)
- **Frontend**: HTML5, CSS3, [Tailwind CSS](https://tailwindcss.com/), [FontAwesome](https://fontawesome.com/), QRCode.js
- **Reporting**: Python CSV Engine & [ReportLab](https://www.reportlab.com/)

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10 or higher installed. Check with:
```bash
python --version
```

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/SmartAttend-AI.git
cd SmartAttend-AI
```

### 2. Create and Activate Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

### Start the Application Server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Once started, open your web browser to:

| Portal | URL | Description |
| :--- | :--- | :--- |
| 👨‍🏫 **Faculty Dashboard** | [http://localhost:8000/faculty](http://localhost:8000/faculty) | Initialize sessions, generate QR code, monitor live roster & export CSV |
| 📝 **Student Registration** | [http://localhost:8000/register](http://localhost:8000/register) | Enroll new students and record baseline facial embedding |
| 📱 **Student Attendance Portal** | [http://localhost:8000/student](http://localhost:8000/student) | Student verification portal (GPS + Liveness + Face match) |

---

## 📖 Step-by-Step Usage

### 1. Register Students
- Navigate to `/register`.
- Enter **Student ID** (e.g. `2025MCA53`), **Student Name**, and **Department**.
- Allow camera access, position face within frame, and click **Register Student**.
- The system computes and stores the 128D baseline embedding in the database.

### 2. Start a Faculty Session
- Navigate to `/faculty`.
- Allow location/GPS permission so the instructor coordinates are locked.
- Enter **Course Name** (e.g. `Computer Networks`), select **Period** (e.g. `Period 2`), enter **Faculty ID/Name**, and select **Geofence Radius** (default `30m`).
- Click **1-Click Start Session**. A QR code will be generated immediately.

### 3. Verify Attendance
- Students in the classroom scan the QR code with their mobile phone (or navigate to `/student?session_id=<UUID>`).
- The student enters their Student ID.
- The system evaluates:
  1. **GPS Distance** to classroom device.
  2. **Live 3D FaceMesh liveness** challenge.
  3. **Biometric facial match**.
- Upon success, the student is marked `PRESENT` and the instructor's live dashboard updates automatically.

### 4. Export Reports
- Click **CSV** on the active session roster to download that period's attendance sheet.
- Switch to the **Master Directory** tab and click **Master CSV** to download cumulative institutional records.

---

## 📁 Project Structure

```
attendmini/
├── main.py                                  # FastAPI application, REST endpoints, computer vision math
├── database.py                              # SQLAlchemy database schema and models
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignored files and directories
├── README.md                                # Project documentation
├── templates/
│   ├── faculty.html                         # Instructor session control & real-time roster dashboard
│   ├── index.html                           # Student mobile verification portal
│   └── register.html                        # Student baseline face enrollment portal
└── SmartAttend_Project_Documentation.docx   # Academic documentation report (.docx)
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
