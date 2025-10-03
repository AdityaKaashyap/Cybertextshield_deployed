from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from app.core.config import settings

# HTTPBearer parses the `Authorization: Bearer <token>` header
security = HTTPBearer()

def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Decode JWT token and return the user's phone number (sub claim).
    Raises HTTP 401 if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(
            token.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload["sub"]  # phone_number from login_user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
