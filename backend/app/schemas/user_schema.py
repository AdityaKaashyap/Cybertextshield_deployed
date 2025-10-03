from pydantic import BaseModel, Field

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    phone_number: str = Field(..., pattern=r"^\d{4,15}$")  # relaxed validation
    password: str = Field(..., min_length=6)

class UserLoginRequest(BaseModel):
    phone_number: str  # <-- login with phone number
    password: str

class UserResponse(BaseModel):
    username: str
    phone_number: str
    message: str

class UserLoginResponse(BaseModel):
    access_token: str
    token_type: str
