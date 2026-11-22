import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, UniqueConstraint, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./smartattend.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False)
    face_embedding = Column(Text, nullable=False)  # JSON-encoded array of floats
    created_at = Column(DateTime, default=datetime.utcnow)

    attendance_logs = relationship("AttendanceLog", back_populates="student", cascade="all, delete-orphan")


class ClassSession(Base):
    __tablename__ = "class_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    course_name = Column(String(150), nullable=False)
    period_name = Column(String(100), default="Period 1")
    faculty_id = Column(String(50), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    geofence_radius = Column(Float, default=30.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    attendance_logs = relationship("AttendanceLog", back_populates="session", cascade="all, delete-orphan")


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), ForeignKey("class_sessions.session_id"), nullable=False, index=True)
    student_id = Column(String(50), ForeignKey("students.student_id"), nullable=False, index=True)
    verified_at = Column(DateTime, default=datetime.utcnow)
    similarity_score = Column(Float, nullable=False)
    distance_meters = Column(Float, nullable=False)
    status = Column(String(20), default="PRESENT")

    session = relationship("ClassSession", back_populates="attendance_logs")
    student = relationship("Student", back_populates="attendance_logs")

    __table_args__ = (
        UniqueConstraint("session_id", "student_id", name="uq_session_student"),
    )


from sqlalchemy import text

def init_db():
    Base.metadata.create_all(bind=engine)
    # Migration checks for columns in existing databases
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE class_sessions ADD COLUMN geofence_radius FLOAT DEFAULT 30.0"))
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE class_sessions ADD COLUMN period_name VARCHAR(100) DEFAULT 'Period 1'"))
            conn.commit()
        except Exception:
            pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
