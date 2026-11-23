import os
from pydantic import BaseModel


class Settings(BaseModel):
    APP_NAME: str = "Smart Dairy Supply Chain Optimization API"
    APP_VERSION: str = "2.0.0"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./dairy_supply_chain.db")
    MODEL_ARTIFACT_PATH: str = os.getenv("MODEL_ARTIFACT_PATH", "./models_store/dairy_ml_bundle.joblib")
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
