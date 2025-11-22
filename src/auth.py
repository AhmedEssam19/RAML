import importlib
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

SECRET_KEY = "WE-CHANGE-THIS-IN-PRODUCTION"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

security = HTTPBearer(auto_error=False)

_jwt_encode = None
_jwt_decode = None

try:
    _jwt = importlib.import_module("jwt")
    if hasattr(_jwt, "encode") and hasattr(_jwt, "decode"):
        _jwt_encode = _jwt.encode
        _jwt_decode = _jwt.decode
    else:
        _jwt = None
except Exception:
    _jwt = None



if _jwt is None:
    try:
        jose_jwt = importlib.import_module("jose.jwt")
        _jwt_encode = jose_jwt.encode
        _jwt_decode = jose_jwt.decode
        _using_jose = True
    except Exception:
        raise ImportError(
            "No suitable JWT library found. Install 'PyJWT' (pip install pyjwt) or 'python-jose' (pip install python-jose)."
        )
else:
    _using_jose = False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    if not _using_jose:
        return _jwt_encode(to_encode, SECRET_KEY, algorithm=ALGORITHM) # type: ignore
    else:
        return _jwt_encode(to_encode, SECRET_KEY, ALGORITHM) # type: ignore


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})

    if not _using_jose:
        return _jwt_encode(to_encode, SECRET_KEY, algorithm=ALGORITHM) # type: ignore
    else:
        return _jwt_encode(to_encode, SECRET_KEY, ALGORITHM) # type: ignore


def verify_token(token: str) -> dict:
    """Verify and decode a JWT token."""
    try:
        if not _using_jose:
            payload = _jwt_decode(token, SECRET_KEY, algorithms=[ALGORITHM]) # type: ignore
        else:
            payload = _jwt_decode(token, SECRET_KEY, algorithms=[ALGORITHM]) # type: ignore
        return payload
    except Exception as e:
        msg = str(e)
        if "expired" in msg.lower() or "signature has expired" in msg.lower():
            raise HTTPException(status_code=401, detail="Token has expired")
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """Dependency to get the current user from JWT token.

    This supports tokens provided via the `Authorization: Bearer ...` header OR
    via an `access_token` httpOnly cookie (preferred for browsers).
    """
    token = None

    if credentials and getattr(credentials, "credentials", None):
        token = credentials.credentials
    else:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = verify_token(token)

    if payload.get("type") == "refresh":
        raise HTTPException(status_code=401, detail="Cannot use refresh token for access")

    username = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {"username": username, "user_id": payload.get("user_id")}
