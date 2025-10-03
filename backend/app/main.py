from fastapi import FastAPI
from app.api.v1 import routes_predict, routes_health, routes_users

app = FastAPI(title="Smishing Detection API", version="1.0")

# Include routers
app.include_router(routes_health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(routes_predict.router, prefix="/api/v1/predict", tags=["prediction"])
app.include_router(routes_users.router, prefix="/api/v1/users", tags=["users"])
