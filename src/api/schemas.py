from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    monthly_income: float
    location: str
    lifestyle: Optional[str] = "student"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TransactionCreate(BaseModel):
    amount: float
    category: str
    type: str
    description: Optional[str] = ""
    date: Optional[datetime] = None

class TransactionResponse(BaseModel):
    id: int
    amount: float
    category: str
    type: str
    description: str
    date: datetime
    
    class Config:
        from_attributes = True

class AlertResponse(BaseModel):
    id: int
    message: str
    severity: str
    is_read: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class DashboardResponse(BaseModel):
    total_income: float
    total_expenses: float
    balance: float
    stability_score: float
    predicted_balance: float
    category_breakdown: dict


class BudgetCreate(BaseModel):
    name: str
    category: Optional[str] = None
    amount: float
    month: Optional[str] = None


class GoalCreate(BaseModel):
    name: str
    target_amount: float
    target_date: datetime


class GoalContributionCreate(BaseModel):
    amount: float


class FeedbackCreate(BaseModel):
    rating: int
    message: str
