from pydantic import BaseModel
from datetime import datetime

class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    email: str
    password: str


class UserLogin(UserBase):
    password: str


class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True


class APKReportCreate(BaseModel):
    user_email: str
    apk_filename: str
    task_id: str


class APKReportUpdate(BaseModel):
    status: str = None
    markdown_report: str = None


class APKReportResponse(BaseModel):
    id: int
    user_email: str
    apk_filename: str
    task_id: str
    status: str
    markdown_report: str = None
    created_at: datetime

    class Config:
        from_attributes = True
