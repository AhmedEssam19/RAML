from sqlalchemy.orm import Session
from fastapi import HTTPException
from passlib.context import CryptContext
from datetime import timedelta
from .. import models, schemas
from ..auth import create_access_token, create_refresh_token, ACCESS_TOKEN_EXPIRE_MINUTES

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def register_user(user: schemas.UserCreate, db: Session):
    if(".com" in user.username):
        raise HTTPException(status_code=400, detail="Username cannot be an email")
    existing_user = db.query(models.User).filter(models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    existing_email = db.query(models.User).filter(models.User.email == user.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    password_bytes = user.password.encode("utf-8")[:72]
    password = password_bytes.decode("utf-8", errors="ignore")
    hashed_pw = pwd_context.hash(password)
    new_user = models.User(username=user.username, email = user.email, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "Registration successful"}


def login_user(user: schemas.UserLogin, db: Session):
    identifier = user.username
    if ".com" in identifier:
        db_user = db.query(models.User).filter(models.User.email == identifier).first()
        not_found_msg = "Email not found"
    else:
        db_user = db.query(models.User).filter(models.User.username == identifier).first()
        not_found_msg = "Username not found"

    if not db_user:
        raise HTTPException(status_code=404, detail=not_found_msg)

    if not pwd_context.verify(user.password, str(db_user.hashed_password)):
        raise HTTPException(status_code=401, detail="Incorrect password")

    access_token = create_access_token(
        data={"sub": db_user.username, "user_id": db_user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = create_refresh_token(
        data={"sub": db_user.username, "user_id": db_user.id}
    )

    return {
        "message": "Login successful",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": db_user.id,
            "username": db_user.username,
            "email": db_user.email
        }
    }
