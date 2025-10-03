import bcrypt
import jwt
import datetime
from fastapi import HTTPException
from app.core.database import users_collection
from app.schemas.user_schema import (
    UserRegisterRequest,
    UserResponse,
    UserLoginRequest,
    UserLoginResponse
)

SECRET_KEY = "ADGP2026"   # ⚠️ store in .env in production
ALGORITHM = "HS256"


# -----------------------
# Password helpers
# -----------------------
def hash_password(password: str) -> str:
    """Hash password securely."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

def create_access_token(data: dict, expires_delta: int = 30) -> str:
    """Create JWT token."""
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_delta)
    data.update({"exp": expire})
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


# -----------------------
# Register User
# -----------------------
async def register_user(request: UserRegisterRequest) -> UserResponse:
    # Check if username OR phone already exists
    existing_user = await users_collection.find_one(
        {"$or": [{"username": request.username}, {"phone_number": request.phone_number}]}
    )
    if existing_user:
        raise HTTPException(status_code=400, detail="User with username or phone already exists")

    hashed_pw = hash_password(request.password)
    new_user = {
        "username": request.username,
        "phone_number": request.phone_number,
        "password": hashed_pw,
    }
    await users_collection.insert_one(new_user)

    return UserResponse(
        username=request.username,
        phone_number=request.phone_number,
        message="User registered successfully"
    )


# -----------------------
# Login User
# -----------------------
async def login_user(request: UserLoginRequest) -> UserLoginResponse:
    user = await users_collection.find_one({"phone_number": request.phone_number})
    if not user or not verify_password(request.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid phone number or password")

    token = create_access_token({"sub": request.phone_number})
    return UserLoginResponse(access_token=token, token_type="bearer")

async def get_user_by_phone(phone_number: str):
    return await users_collection.find_one({"phone_number": phone_number})
