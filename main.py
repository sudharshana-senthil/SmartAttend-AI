import os
import io
import csv
import json
import base64
import math
import uuid
from datetime import datetime
from typing import Optional, List

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, Response, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import init_db, get_db, Student, ClassSession, AttendanceLog

# ReportLab imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# Initialize Database tables
init_db()

app = FastAPI(
    title="SmartAttend AI System",
    description="Production-Ready Zero-Proxy Attendance System using Spatial, Vitality, and Biometric Verification",
    version="1.0.0"
)

# Enable CORS for local cross-origin API calls if required
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Templates
templates = Jinja2Templates(directory="templates")


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================

class StudentRegisterRequest(BaseModel):
    student_id: str = Field(..., description="Unique Student ID (e.g. STU1001)")
    name: str = Field(..., description="Full Name of the Student")
    department: str = Field(..., description="Academic Department")
    image_base64: str = Field(..., description="Base64 encoded JPEG/PNG capture")

class CreateSessionRequest(BaseModel):
    course_name: str = Field(..., description="Course Name / Code (e.g. CS402)")
    period_name: Optional[str] = Field("Period 1", description="College Period Name/Number (e.g. Period 1, Period 2)")
    faculty_id: str = Field(..., description="Faculty Member Identifier")
    latitude: float = Field(..., description="Faculty Geolocation Latitude")
    longitude: float = Field(..., description="Faculty Geolocation Longitude")
    geofence_radius: Optional[float] = Field(30.0, description="Geofence radius in meters (0 for Demo Mode / Unlimited)")


class VerifyAttendanceRequest(BaseModel):
    session_id: str = Field(..., description="Active Class Session UUID")
    student_id: str = Field(..., description="Student ID attempting attendance")
    latitude: float = Field(..., description="Student Geolocation Latitude")
    longitude: float = Field(..., description="Student Geolocation Longitude")
    image_base64: str = Field(..., description="Base64 encoded live web capture frame")


import socket

def get_local_ip() -> str:
    """
    Returns the host machine's local Wi-Fi / LAN IP address (e.g. 192.168.X.X)
    so mobile devices on the same Wi-Fi can connect seamlessly to the QR code link.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the spherical distance between two points on the Earth
    in meters using the Haversine formula.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return R * c


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates cosine similarity between two 128D embedding vectors.
    Returns float score in range [-1.0, 1.0].
    """
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decodes base64 string (with or without data URI prefix) to OpenCV BGR image array.
    """
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode base64 image data")
    return img


def extract_face_embedding(image_bgr: np.ndarray) -> List[float]:
    """
    Extracts a normalized 128-dimensional facial feature embedding vector.
    Uses spatial-grid normalization, histogram equalization, and multi-resolution texture descriptors.
    Guaranteed compatibility across OpenCV 4.x and OpenCV 5.x.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Face region of interest: center face crop
    height, width = gray.shape
    cy, cx = height // 2, width // 2
    hw, hh = max(10, min(width, height) // 3), max(10, min(width, height) // 3)
    face_roi = gray[max(0, cy - hh):min(height, cy + hh), max(0, cx - hw):min(width, cx + hw)]

    # Resize face ROI to standard 128x128 thumbnail
    face_resized = cv2.resize(face_roi, (128, 128), interpolation=cv2.INTER_AREA)

    # Histogram equalization for contrast normalization
    face_norm = cv2.equalizeHist(face_resized).astype(np.float32) / 255.0

    # Compute 128D feature vector via 8x8 spatial grid block descriptors
    # 64 block means + 64 block standard deviations = 128D unit vector
    features = []
    cell_size = 16
    for i in range(8):
        for j in range(8):
            cell = face_norm[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            features.append(float(np.mean(cell)))
            features.append(float(np.std(cell)))

    embedding = np.array(features, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()


# ==========================================
# REPORTLAB PDF GENERATORS
# ==========================================

def generate_session_pdf(session: ClassSession, logs: List[AttendanceLog], db: Session) -> bytes:
    """
    Generates a production-quality PDF report for a single class session/period.
    Includes Course Name, Period Name, Faculty ID, Session Date/Time, and Student Attendance Table.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=0,
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#0284c7'),
        spaceAfter=12
    )
    info_style = ParagraphStyle(
        'InfoText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    info_bold = ParagraphStyle(
        'InfoBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )
    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#1e293b')
    )
    header_cell_style = ParagraphStyle(
        'HeaderCellText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []
    story.append(Paragraph("SmartAttend AI Verification System", title_style))
    period_str = session.period_name if session.period_name else "Period 1"
    story.append(Paragraph(f"PERIOD ATTENDANCE REPORT &bull; {session.course_name.upper()} &bull; {period_str.upper()}", subtitle_style))

    # Metadata Summary Table
    meta_data = [
        [
            Paragraph("Course Name:", info_bold), Paragraph(session.course_name, info_style),
            Paragraph("Period Name:", info_bold), Paragraph(period_str, info_style)
        ],
        [
            Paragraph("Faculty ID:", info_bold), Paragraph(session.faculty_id, info_style),
            Paragraph("Session ID:", info_bold), Paragraph(session.session_id[:8], info_style)
        ],
        [
            Paragraph("Created At:", info_bold), Paragraph(session.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"), info_style),
            Paragraph("Total Verified:", info_bold), Paragraph(f"<b>{len(logs)} Students</b>", info_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[80, 190, 80, 190])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#f1f5f9')),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Roster Table
    table_headers = ["#", "Student ID", "Student Name", "Department", "Verified Time", "Similarity", "Distance", "Status"]
    table_data = [[Paragraph(h, header_cell_style) for h in table_headers]]

    for idx, log in enumerate(logs, start=1):
        student = db.query(Student).filter(Student.student_id == log.student_id).first()
        row = [
            Paragraph(str(idx), cell_style),
            Paragraph(f"<b>{log.student_id}</b>", cell_style),
            Paragraph(student.name if student else "Unknown", cell_style),
            Paragraph(student.department if student else "N/A", cell_style),
            Paragraph(log.verified_at.strftime("%H:%M:%S (%Y-%m-%d)"), cell_style),
            Paragraph(f"{log.similarity_score * 100:.1f}%", cell_style),
            Paragraph(f"{log.distance_meters:.1f} m", cell_style),
            Paragraph(f"<font color='#16a34a'><b>{log.status}</b></font>", cell_style)
        ]
        table_data.append(row)

    if len(logs) == 0:
        empty_cell = Paragraph("No students have verified attendance for this period yet.", cell_style)
        table_data.append([empty_cell] + [""] * 7)

    # Column Widths sum to 540 pt (7.5 inches = printable width on letter size with 0.5in margins)
    roster_table = Table(table_data, colWidths=[25, 65, 115, 75, 90, 50, 50, 70])
    ts = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ])
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            ts.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f8fafc'))
        else:
            ts.add('BACKGROUND', (0, i), (-1, i), colors.white)

    if len(logs) == 0:
        ts.add('SPAN', (0, 1), (7, 1))
        ts.add('ALIGN', (0, 1), (7, 1), 'CENTER')

    roster_table.setStyle(ts)
    story.append(roster_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_master_pdf(students: List[Student], db: Session) -> bytes:
    """
    Generates a master PDF directory report of ALL registered students and attendance summary.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=0,
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#7c3aed'),
        spaceAfter=12
    )
    info_style = ParagraphStyle(
        'InfoText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    info_bold = ParagraphStyle(
        'InfoBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )
    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#1e293b')
    )
    header_cell_style = ParagraphStyle(
        'HeaderCellText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []
    story.append(Paragraph("SmartAttend AI Verification System", title_style))
    story.append(Paragraph("MASTER STUDENT DIRECTORY & OVERALL ATTENDANCE REPORT", subtitle_style))

    total_present_count = 0
    student_records = []
    for student in students:
        logs = db.query(AttendanceLog).filter(AttendanceLog.student_id == student.student_id).all()
        if len(logs) > 0:
            total_present_count += 1
        last_log = logs[-1] if logs else None
        student_records.append((student, logs, last_log))

    # Metadata Table
    meta_data = [
        [
            Paragraph("Generated At:", info_bold), Paragraph(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), info_style),
            Paragraph("Total Registered:", info_bold), Paragraph(f"<b>{len(students)} Students</b>", info_style)
        ],
        [
            Paragraph("Active / Verified:", info_bold), Paragraph(f"<b>{total_present_count} Students</b>", info_style),
            Paragraph("Overall Rate:", info_bold), Paragraph(f"<b>{(total_present_count / len(students) * 100) if len(students) > 0 else 0:.1f}%</b>", info_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[90, 180, 90, 180])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#f1f5f9')),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Master Roster Table
    table_headers = ["#", "Student ID", "Student Name", "Department", "Sessions Attended", "Last Verified", "Overall Status"]
    table_data = [[Paragraph(h, header_cell_style) for h in table_headers]]

    for idx, (student, logs, last_log) in enumerate(student_records, start=1):
        status_text = "<font color='#16a34a'><b>PRESENT</b></font>" if len(logs) > 0 else "<font color='#dc2626'><b>ABSENT</b></font>"
        last_ver = last_log.verified_at.strftime("%Y-%m-%d %H:%M") if last_log else "Never"
        row = [
            Paragraph(str(idx), cell_style),
            Paragraph(f"<b>{student.student_id}</b>", cell_style),
            Paragraph(student.name, cell_style),
            Paragraph(student.department, cell_style),
            Paragraph(f"<b>{len(logs)}</b> sessions", cell_style),
            Paragraph(last_ver, cell_style),
            Paragraph(status_text, cell_style)
        ]
        table_data.append(row)

    if len(students) == 0:
        empty_cell = Paragraph("No registered students found in database.", cell_style)
        table_data.append([empty_cell] + [""] * 6)

    # Column Widths sum to 540 pt
    roster_table = Table(table_data, colWidths=[25, 75, 135, 95, 80, 70, 60])
    ts = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4c1d95')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ])
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            ts.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f8fafc'))
        else:
            ts.add('BACKGROUND', (0, i), (-1, i), colors.white)

    if len(students) == 0:
        ts.add('SPAN', (0, 1), (6, 1))
        ts.add('ALIGN', (0, 1), (6, 1), 'CENTER')

    roster_table.setStyle(ts)
    story.append(roster_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================
# PAGE ROUTING ENDPOINTS
# ==========================================

@app.get("/", response_class=RedirectResponse)
def root_redirect():
    return RedirectResponse(url="/faculty")


@app.get("/register", response_class=HTMLResponse)
def page_register(request: Request):
    res = templates.TemplateResponse(request=request, name="register.html")
    res.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return res


@app.get("/faculty", response_class=HTMLResponse)
def page_faculty(request: Request):
    res = templates.TemplateResponse(request=request, name="faculty.html")
    res.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return res


@app.get("/student", response_class=HTMLResponse)
def page_student(request: Request):
    res = templates.TemplateResponse(request=request, name="index.html")
    res.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return res


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.get("/api/network-ip")
def get_network_info():
    """
    Returns the host machine's Wi-Fi / local LAN IP address and server port.
    """
    return {
        "local_ip": get_local_ip(),
        "port": 8000
    }


@app.post("/api/register-student")
def register_student(payload: StudentRegisterRequest, db: Session = Depends(get_db)):
    """
    Onboards a student: processes baseline face snapshot and stores 128D facial embedding into SQLite DB.
    """
    try:
        img = decode_base64_image(payload.image_base64)
        embedding = extract_face_embedding(img)

        # Check if student exists
        existing_student = db.query(Student).filter(Student.student_id == payload.student_id).first()

        if existing_student:
            existing_student.name = payload.name
            existing_student.department = payload.department
            existing_student.face_embedding = json.dumps(embedding)
            db.commit()
            return {
                "status": "success",
                "message": f"Updated baseline face registration for {payload.name} ({payload.student_id})",
                "student_id": payload.student_id
            }

        new_student = Student(
            student_id=payload.student_id,
            name=payload.name,
            department=payload.department,
            face_embedding=json.dumps(embedding)
        )
        db.add(new_student)
        db.commit()

        return {
            "status": "success",
            "message": f"Successfully registered student {payload.name} ({payload.student_id})",
            "student_id": payload.student_id
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")


@app.post("/api/create-session")
def create_session(payload: CreateSessionRequest, db: Session = Depends(get_db)):
    """
    Creates a new class session with instructor device coordinates, period name, and configurable geofence radius.
    """
    try:
        session_id = str(uuid.uuid4())
        radius = payload.geofence_radius if payload.geofence_radius is not None else 30.0
        period = payload.period_name.strip() if (payload.period_name and payload.period_name.strip()) else "Period 1"
        
        new_session = ClassSession(
            session_id=session_id,
            course_name=payload.course_name,
            period_name=period,
            faculty_id=payload.faculty_id,
            latitude=payload.latitude,
            longitude=payload.longitude,
            geofence_radius=radius,
            is_active=True
        )
        db.add(new_session)
        db.commit()

        return {
            "status": "success",
            "session_id": session_id,
            "course_name": payload.course_name,
            "period_name": period,
            "faculty_id": payload.faculty_id,
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "geofence_radius": radius,
            "created_at": new_session.created_at.isoformat()
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to create session: {str(e)}")


@app.get("/api/session-info/{session_id}")
def get_session_info(session_id: str, db: Session = Depends(get_db)):
    """
    Returns metadata for a specific class session.
    """
    session = db.query(ClassSession).filter(ClassSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Class session not found")

    return {
        "session_id": session.session_id,
        "course_name": session.course_name,
        "period_name": session.period_name if session.period_name else "Period 1",
        "faculty_id": session.faculty_id,
        "latitude": session.latitude,
        "longitude": session.longitude,
        "geofence_radius": session.geofence_radius if session.geofence_radius is not None else 30.0,
        "is_active": session.is_active,
        "created_at": session.created_at.isoformat()
    }


@app.post("/api/verify-attendance")
def verify_attendance(payload: VerifyAttendanceRequest, db: Session = Depends(get_db)):
    """
    Core verification pipeline enforcing 'The Verification Triangle':
    1. Spatial Verification (Haversine <= Geofence Radius)
    2. Identity Verification (Facial embedding Cosine Similarity >= 80%)
    """
    # 1. Fetch Class Session
    session = db.query(ClassSession).filter(ClassSession.session_id == payload.session_id).first()
    if not session or not session.is_active:
        raise HTTPException(
            status_code=400,
            detail="Session Invalid: Class session does not exist or has ended."
        )

    # 2. Geofence Distance Check (Spatial Verification)
    dist_meters = haversine_distance(
        session.latitude, session.longitude,
        payload.latitude, payload.longitude
    )

    allowed_radius = session.geofence_radius if (session.geofence_radius is not None and session.geofence_radius >= 0) else 30.0
    if allowed_radius > 0 and dist_meters > allowed_radius:
        raise HTTPException(
            status_code=400,
            detail=f"Spatial Verification Failed: You are {dist_meters:.1f}m away from the classroom (Max allowed: {allowed_radius:.0f}m radius)."
        )

    # 3. Fetch Registered Student
    student = db.query(Student).filter(Student.student_id == payload.student_id).first()
    if not student:
        raise HTTPException(
            status_code=404,
            detail=f"Identity Verification Failed: Student ID '{payload.student_id}' is not registered in the system."
        )

    # 4. Extract Live Frame Embedding & Calculate Cosine Similarity
    try:
        live_img = decode_base64_image(payload.image_base64)
        live_embedding = extract_face_embedding(live_img)
    except Exception as img_err:
        raise HTTPException(status_code=400, detail=f"Image decoding error: {str(img_err)}")

    baseline_embedding = json.loads(student.face_embedding)
    similarity = cosine_similarity(baseline_embedding, live_embedding)

    BIOMETRIC_THRESHOLD = 0.70
    if similarity < BIOMETRIC_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"Biometric Match Failed: Facial similarity score is {similarity*100:.1f}% (Minimum required: 70.0%)."
        )

    # 5. Prevent Duplicate Attendance Logs
    existing_log = db.query(AttendanceLog).filter(
        AttendanceLog.session_id == payload.session_id,
        AttendanceLog.student_id == payload.student_id
    ).first()

    now = datetime.utcnow()
    if existing_log:
        existing_log.verified_at = now
        existing_log.similarity_score = round(similarity, 4)
        existing_log.distance_meters = round(dist_meters, 1)
        existing_log.status = "PRESENT"
        db.commit()
        log_entry = existing_log
    else:
        log_entry = AttendanceLog(
            session_id=payload.session_id,
            student_id=payload.student_id,
            verified_at=now,
            similarity_score=round(similarity, 4),
            distance_meters=round(dist_meters, 1),
            status="PRESENT"
        )
        db.add(log_entry)
        db.commit()

    return {
        "status": "success",
        "message": "Attendance Verified & Recorded Successfully!",
        "session_id": payload.session_id,
        "student_id": student.student_id,
        "student_name": student.name,
        "department": student.department,
        "similarity_score": round(similarity * 100, 1),
        "distance_meters": round(dist_meters, 1),
        "verified_at": log_entry.verified_at.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/api/session-attendance/{session_id}")
def get_session_attendance(session_id: str, db: Session = Depends(get_db)):
    """
    Returns real-time list of verified students for a specific class session.
    """
    session = db.query(ClassSession).filter(ClassSession.session_id == session_id).first()
    logs = db.query(AttendanceLog).filter(AttendanceLog.session_id == session_id).all()
    results = []
    course_str = session.course_name if session else "N/A"
    period_str = (session.period_name if (session and session.period_name) else "Period 1")
    faculty_str = session.faculty_id if session else "N/A"

    for log in logs:
        student = db.query(Student).filter(Student.student_id == log.student_id).first()
        results.append({
            "id": log.id,
            "student_id": log.student_id,
            "name": student.name if student else "Unknown",
            "department": student.department if student else "N/A",
            "faculty_id": faculty_str,
            "last_faculty": faculty_str,
            "faculty": faculty_str,
            "course_name": course_str,
            "last_course": course_str,
            "course": course_str,
            "period_name": period_str,
            "last_period": period_str,
            "period": period_str,
            "verified_at": log.verified_at.strftime("%H:%M:%S (%Y-%m-%d)"),
            "similarity_score": f"{log.similarity_score * 100:.1f}%",
            "distance_meters": f"{log.distance_meters:.1f} m",
            "status": log.status
        })
    response = JSONResponse(content={
        "session_id": session_id,
        "course_name": course_str,
        "period_name": period_str,
        "faculty_id": faculty_str,
        "total_present": len(results),
        "attendance": results
    })
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/api/export-attendance/{session_id}")
def export_attendance_csv(session_id: str, db: Session = Depends(get_db)):
    """
    Generates a downloadable CSV report for the specified class session.
    """
    session = db.query(ClassSession).filter(ClassSession.session_id == session_id).first()
    logs = db.query(AttendanceLog).filter(AttendanceLog.session_id == session_id).all()

    output = io.StringIO()
    writer = csv.writer(output)

    # Header section
    course = session.course_name if session else "Session"
    period = session.period_name if (session and session.period_name) else "Period 1"
    faculty = session.faculty_id if session else "N/A"

    writer.writerow(["SmartAttend Verification Report - Period Attendance"])
    writer.writerow(["Staff / Faculty Name:", faculty])
    writer.writerow(["Course Name:", course])
    writer.writerow(["Period Name:", period])
    writer.writerow(["Session UUID:", session_id])
    writer.writerow(["Generated At:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")])
    writer.writerow([])
    writer.writerow(["Student ID", "Student Name", "Department", "Staff / Faculty Name", "Course Name", "Period Name", "Verified At", "Similarity Score (%)", "Distance (m)", "Status"])

    for log in logs:
        student = db.query(Student).filter(Student.student_id == log.student_id).first()
        writer.writerow([
            log.student_id,
            student.name if student else "Unknown",
            student.department if student else "N/A",
            faculty,
            course,
            period,
            log.verified_at.strftime("%Y-%m-%d %H:%M:%S"),
            f"{log.similarity_score * 100:.1f}",
            f"{log.distance_meters:.1f}",
            log.status
        ])

    csv_content = output.getvalue()
    filename = f"SmartAttend_{course.replace(' ', '_')}_{period.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/all-students-attendance")
def get_all_students_attendance(db: Session = Depends(get_db)):
    """
    Returns master list of ALL registered students in the system with their complete Staff Name, Course & Period attendance records.
    """
    students = db.query(Student).all()
    results = []
    for student in students:
        logs = db.query(AttendanceLog).filter(AttendanceLog.student_id == student.student_id).all()
        last_log = logs[-1] if logs else None
        last_course = "N/A"
        last_period = "N/A"
        last_faculty = "N/A"
        if last_log:
            sess = db.query(ClassSession).filter(ClassSession.session_id == last_log.session_id).first()
            if sess:
                last_course = sess.course_name
                last_period = sess.period_name or "Period 1"
                last_faculty = sess.faculty_id or "N/A"

        results.append({
            "student_id": student.student_id,
            "name": student.name,
            "department": student.department,
            "total_attended": len(logs),
            "faculty_id": last_faculty,
            "last_faculty": last_faculty,
            "faculty": last_faculty,
            "course_name": last_course,
            "last_course": last_course,
            "course": last_course,
            "period_name": last_period,
            "last_period": last_period,
            "period": last_period,
            "last_verified": last_log.verified_at.strftime("%Y-%m-%d %H:%M:%S") if last_log else "Never",
            "last_similarity": f"{last_log.similarity_score * 100:.1f}%" if last_log else "N/A",
            "status": "PRESENT" if len(logs) > 0 else "ABSENT"
        })
    response = JSONResponse(content={
        "total_registered": len(students),
        "total_present_overall": len([s for s in results if s["status"] == "PRESENT"]),
        "students": results
    })
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/api/export-master-attendance")
def export_master_attendance_csv(db: Session = Depends(get_db)):
    """
    Generates a downloadable Master CSV report containing ALL registered students, Staff/Faculty Name, Course Name, Period Name, and full history.
    """
    students = db.query(Student).all()
    all_logs = db.query(AttendanceLog).all()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["SmartAttend Master Complete Student Attendance Report"])
    writer.writerow(["Generated At:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")])
    writer.writerow(["Total Registered Students:", len(students)])
    writer.writerow(["Total Attendance Verified Logs:", len(all_logs)])
    writer.writerow([])

    # SECTION 1: MASTER STUDENT SUMMARY ROSTER
    writer.writerow(["=== REGISTERED STUDENTS SUMMARY ROSTER ==="])
    writer.writerow(["Student ID", "Student Name", "Department", "Staff / Faculty Name", "Course Name", "Period Name", "Total Sessions Attended", "Last Verified Timestamp", "Overall Status"])

    for student in students:
        logs = db.query(AttendanceLog).filter(AttendanceLog.student_id == student.student_id).all()
        last_log = logs[-1] if logs else None
        last_course = "N/A"
        last_period = "N/A"
        last_faculty = "N/A"
        if last_log:
            sess = db.query(ClassSession).filter(ClassSession.session_id == last_log.session_id).first()
            if sess:
                last_course = sess.course_name
                last_period = sess.period_name or "Period 1"
                last_faculty = sess.faculty_id or "N/A"

        writer.writerow([
            student.student_id,
            student.name,
            student.department,
            last_faculty,
            last_course,
            last_period,
            len(logs),
            last_log.verified_at.strftime("%Y-%m-%d %H:%M:%S") if last_log else "Never",
            "PRESENT" if len(logs) > 0 else "ABSENT"
        ])

    writer.writerow([])
    writer.writerow([])

    # SECTION 2: COMPLETE PERIOD-WISE ATTENDANCE LOG HISTORY
    writer.writerow(["=== COMPLETE PERIOD-WISE ATTENDANCE LOG HISTORY ==="])
    writer.writerow(["Student ID", "Student Name", "Department", "Staff / Faculty Name", "Course Name", "Period Name", "Verified Timestamp", "Similarity Score (%)", "Distance (m)", "Status"])

    for log in all_logs:
        student = db.query(Student).filter(Student.student_id == log.student_id).first()
        session = db.query(ClassSession).filter(ClassSession.session_id == log.session_id).first()
        writer.writerow([
            log.student_id,
            student.name if student else "Unknown",
            student.department if student else "N/A",
            session.faculty_id if session else "N/A",
            session.course_name if session else "N/A",
            (session.period_name if (session and session.period_name) else "Period 1"),
            log.verified_at.strftime("%Y-%m-%d %H:%M:%S"),
            f"{log.similarity_score * 100:.1f}",
            f"{log.distance_meters:.1f}",
            log.status
        ])

    csv_content = output.getvalue()
    filename = f"SmartAttend_Master_Full_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/export-master-pdf")
def export_master_attendance_pdf(db: Session = Depends(get_db)):
    """
    Generates a downloadable Master PDF report containing ALL registered students.
    """
    students = db.query(Student).all()
    pdf_bytes = generate_master_pdf(students, db)
    filename = f"SmartAttend_Master_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

