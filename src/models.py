from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from datetime import datetime
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)


class APKReport(Base):
    __tablename__ = "apk_reports"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, ForeignKey("users.email"), index=True)
    apk_filename = Column(String)
    task_id = Column(String, unique=True, index=True)
    status = Column(String, default="Started")  # Started, In Progress, Completed, Failed
    markdown_report = Column(Text, nullable=True)  # Stores markdown content or null if not ready
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
