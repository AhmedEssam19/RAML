from fastapi import APIRouter, Depends, Response, Request
from typing import Optional
from sqlalchemy.orm import Session
from .. import schemas
from ..database import get_db
from ..services import user_service
from ..auth import get_current_user, verify_token, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
from datetime import timedelta

router = APIRouter(prefix="/user", tags=["Users"])


@router.post("/register")
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return user_service.register_user(user, db)


@router.post("/login")
def login_user(user: schemas.UserLogin, response: Response, db: Session = Depends(get_db)):
    """Login the user, set httpOnly cookies for access and refresh tokens.

    For browser clients we set `access_token` and `refresh_token` as httpOnly cookies.
    We return only a minimal JSON (no tokens) since tokens are set as cookies.
    """
    result = user_service.login_user(user, db)
    access = result.get("access_token")
    refresh = result.get("refresh_token")

    if access:
        response.set_cookie(
            key="access_token",
            value=access,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    if refresh:
        response.set_cookie(
            key="refresh_token",
            value=refresh,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
        )

    
    return {
        "message": result.get("message", "Login successful"),
        "user": result.get("user"),
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer" if access else None,
    }


@router.post("/refresh")
def refresh_access_token(request: Request, response: Response, refresh_token: Optional[dict] = None):
    """
    Refresh the access token using a valid refresh token.
    
    Expected body: {"refresh_token": "token_string"}
    """
    token = None
    if request is not None:
        token = request.cookies.get("refresh_token")

    if not token and refresh_token:
        token = refresh_token.get("refresh_token")

    if not token:
        raise ValueError("Refresh token is required")

    payload = verify_token(token)
    
    if payload.get("type") != "refresh":
        raise ValueError("Invalid token type")
    
    new_access_token = create_access_token(
        data={"sub": payload.get("sub"), "user_id": payload.get("user_id")},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    response.set_cookie(
        key="access_token",
        value=new_access_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

    return {"message": "Access token refreshed"}


@router.get("/me")
def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get information about the currently authenticated user."""
    return {
        "username": current_user["username"],
        "user_id": current_user["user_id"]
    }


@router.post("/logout")
def logout(response: Response, current_user: dict = Depends(get_current_user)):
    """Logout endpoint: clear auth cookies."""
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return {"message": "Logged out successfully"}
