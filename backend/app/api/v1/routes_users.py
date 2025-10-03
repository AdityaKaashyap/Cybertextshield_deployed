from fastapi import APIRouter, HTTPException,Depends
from app.schemas.user_schema import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    UserLoginResponse
)
from app.services.user_service import register_user, login_user, get_user_by_phone
from app.core.auth import get_current_user 

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register(request: UserRegisterRequest):
    return await register_user(request)

@router.post("/login", response_model=UserLoginResponse)
async def login(request: UserLoginRequest):
    token_response = await login_user(request)
    if not token_response:
        raise HTTPException(status_code=401, detail="Invalid phone number or password")
    return token_response

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: str = Depends(get_current_user)):
    """
    Get the current logged-in user's profile.
    `current_user` is extracted from JWT (phone_number).
    """
    user = await get_user_by_phone(current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        username=user["username"],
        phone_number=user["phone_number"],
        message="User profile fetched successfully"
    )
