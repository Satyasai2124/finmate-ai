"""Configuration settings for the Financial Stability Support System"""
import os

class Config:
    APP_NAME = "FINMATE AI"
    VERSION = "1.0.0"
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/financial_stability.db")
    SECRET_KEY = os.getenv("SECRET_KEY", "development-only-secret-change-in-production")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 1440
    MIN_TRANSACTIONS_FOR_PREDICTION = 3
    PREDICTION_DAYS_AHEAD = 30
    FOOD_EXPENSE_THRESHOLD = 0.40
    CATEGORY_EXPENSE_THRESHOLD = 0.50
