from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List
import pandas as pd
import os
import tempfile
import requests
import math

from src.api.database import get_db, init_db
from src.api.models import (
    User, Transaction, Alert, Budget, FinancialGoal, GoalContribution, Feedback,
    Loan, Subscription, ShoppingItem, SharedExpenseGroup, SharedGroupMember,
    SharedExpense, ManualSettlement, Bill, SavingsChallenge, CoachMessage
)
from src.api.schemas import *
from src.api.auth import hash_password, verify_password, create_access_token, get_current_user
from src.ml.ml_engine import FinancialMLEngine
from src.api.recommendation_engine import RecommendationEngine
from src.api.spending_predictor import predict_spending
from src.api.config import Config
import csv
import io

app = FastAPI(
    title=Config.APP_NAME,
    version=Config.VERSION,
    description="AI-powered personal financial management system"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
        if Config.ENVIRONMENT == "production"
        else ["*"]
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_engine = FinancialMLEngine()
rec_engine = RecommendationEngine()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def startup():
    if Config.ENVIRONMENT == "production" and Config.SECRET_KEY == "development-only-secret-change-in-production":
        raise RuntimeError("SECRET_KEY must be configured in production")
    init_db()
    db = next(get_db())
    try:
        if Config.ENVIRONMENT == "production":
            return
        demo = db.query(User).filter(User.email == "demo@finova.app").first()
        if not demo:
            demo = User(
                email="demo@finova.app",
                hashed_password=hash_password("Demo@123"),
                name="Demo User",
                monthly_income=50000,
                location="Chennai, India",
                lifestyle="employee"
            )
            db.add(demo)
            db.commit()
            db.refresh(demo)
            now = datetime.utcnow()
            db.add(Transaction(
                user_id=demo.id,
                amount=50000,
                category="salary",
                type="income",
                description="Monthly salary",
                date=now
            ))
            samples = [
                ("food", 8500), ("transport", 4200), ("shopping", 6500),
                ("utilities", 4800), ("entertainment", 3000), ("healthcare", 2300),
                ("education", 5500), ("subscriptions", 2200)
            ]
            for index, (category, amount) in enumerate(samples):
                db.add(Transaction(
                    user_id=demo.id,
                    amount=amount,
                    category=category,
                    type="expense",
                    description=f"Sample {category} expense",
                    date=now - timedelta(days=index * 3)
                ))
            db.commit()
        else:
            demo.lifestyle = "employee"
            db.commit()
    finally:
        db.close()

@app.get("/")
def root():
    return FileResponse('static/index.html')

@app.get("/train")
def train_page():
    return FileResponse('static/train.html')

@app.get("/predict")
def predict_page():
    return FileResponse('static/predict.html')

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": Config.VERSION}

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        name=user_data.name,
        monthly_income=user_data.monthly_income,
        location=user_data.location,
        lifestyle=getattr(user_data, "lifestyle", "student")
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_access_token({"sub": user.email})
    return {
        "access_token": token, 
        "token_type": "bearer",
        "user": {"name": user.name, "email": user.email}
    }

@app.post("/login")
def login(credentials: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == credentials.username).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.email})
    return {
        "access_token": token, 
        "token_type": "bearer",
        "user": {"name": user.name, "email": user.email}
    }

@app.post("/transactions", response_model=TransactionResponse)
def add_transaction(
    transaction: TransactionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    trans = Transaction(
        user_id=current_user.id,
        amount=transaction.amount,
        category=transaction.category,
        type=transaction.type,
        description=transaction.description,
        date=transaction.date or datetime.utcnow()
    )
    db.add(trans)
    db.commit()
    db.refresh(trans)
    
    recent_transactions = db.query(Transaction).filter(
        Transaction.user_id == current_user.id,
        Transaction.date > datetime.utcnow() - timedelta(days=30)
    ).all()
    
    alerts = ml_engine.detect_overspending(current_user, recent_transactions)
    for alert_data in alerts:
        alert = Alert(
            user_id=current_user.id,
            message=alert_data['message'],
            severity=alert_data['severity']
        )
        db.add(alert)
    db.commit()
    
    return trans

@app.get("/transactions", response_model=List[TransactionResponse])
def get_transactions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Transaction).filter(Transaction.user_id == current_user.id).order_by(Transaction.date.desc()).all()

@app.put("/transactions/{transaction_id}", response_model=TransactionResponse)
def update_transaction(
    transaction_id: int,
    transaction: TransactionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    trans = db.query(Transaction).filter(
        Transaction.id == transaction_id,
        Transaction.user_id == current_user.id
    ).first()
    
    if not trans:
        print(f"Transaction {transaction_id} not found for user {current_user.id}")
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
    
    trans.amount = transaction.amount
    trans.category = transaction.category
    trans.type = transaction.type
    trans.description = transaction.description
    
    db.commit()
    db.refresh(trans)
    return trans

@app.delete("/transactions/{transaction_id}")
def delete_transaction(
    transaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    trans = db.query(Transaction).filter(
        Transaction.id == transaction_id,
        Transaction.user_id == current_user.id
    ).first()
    
    if not trans:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    db.delete(trans)
    db.commit()
    return {"message": "Transaction deleted successfully"}

@app.get("/alerts", response_model=List[AlertResponse])
def get_alerts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Alert).filter(Alert.user_id == current_user.id).order_by(Alert.created_at.desc()).all()

@app.delete("/alerts/clear")
def clear_alerts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db.query(Alert).filter(Alert.user_id == current_user.id).delete()
    db.commit()
    return {"message": "All alerts cleared"}

@app.get("/dashboard", response_model=DashboardResponse)
def get_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    transactions = db.query(Transaction).filter(
        Transaction.user_id == current_user.id,
        Transaction.date > datetime.utcnow() - timedelta(days=30)
    ).all()
    
    total_income = sum(t.amount for t in transactions if t.type == 'income')
    total_expenses = sum(t.amount for t in transactions if t.type == 'expense')
    balance = current_user.monthly_income + total_income - total_expenses
    
    stability_score = ml_engine.calculate_stability_score(current_user, transactions)
    predicted_spending = ml_engine.predict_next_month_spending(transactions)
    predicted_balance = current_user.monthly_income - predicted_spending
    
    category_breakdown = ml_engine.analyze_spending_pattern(transactions)
    
    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "balance": balance,
        "stability_score": stability_score,
        "predicted_balance": predicted_balance,
        "category_breakdown": category_breakdown
    }

@app.get("/profile")
def get_profile(
    current_user: User = Depends(get_current_user)
):
    return {
        "name": current_user.name,
        "email": current_user.email,
        "monthly_income": current_user.monthly_income,
        "location": current_user.location,
        "lifestyle": getattr(current_user, "lifestyle", "student"),
        "next_income_date": current_user.next_income_date.isoformat() if current_user.next_income_date else None,
        "emergency_reserve": current_user.emergency_reserve or 0
    }

@app.put("/profile")
def update_profile(
    profile: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    current_user.name = profile.get('name', current_user.name)
    current_user.monthly_income = profile.get('monthly_income', current_user.monthly_income)
    current_user.location = profile.get('location', current_user.location)
    if profile.get('lifestyle') in ('student', 'employee'):
        current_user.lifestyle = profile['lifestyle']
    if profile.get('next_income_date'):
        try:
            current_user.next_income_date = datetime.fromisoformat(profile['next_income_date'].replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid next income date")
    if profile.get('emergency_reserve') is not None:
        current_user.emergency_reserve = max(float(profile['emergency_reserve']), 0)
    
    db.commit()
    db.refresh(current_user)
    
    return {
        "name": current_user.name,
        "email": current_user.email,
        "monthly_income": current_user.monthly_income,
        "location": current_user.location,
        "lifestyle": current_user.lifestyle,
        "next_income_date": current_user.next_income_date.isoformat() if current_user.next_income_date else None,
        "emergency_reserve": current_user.emergency_reserve or 0
    }

@app.get("/recommendations")
def get_recommendations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    transactions = db.query(Transaction).filter(
        Transaction.user_id == current_user.id,
        Transaction.date > datetime.utcnow() - timedelta(days=30)
    ).all()
    
    stability_score = ml_engine.calculate_stability_score(current_user, transactions)
    recommendations = rec_engine.generate_recommendations(current_user, transactions, stability_score)
    
    return {"recommendations": recommendations}


@app.put("/alerts/{alert_id}/read")
def mark_alert_read(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    alert = db.query(Alert).filter(Alert.id == alert_id, Alert.user_id == current_user.id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.is_read = True
    db.commit()
    return {"message": "Alert marked as read"}


@app.get("/budgets")
def get_budgets(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    month = datetime.utcnow().strftime("%Y-%m")
    budgets = db.query(Budget).filter(Budget.user_id == current_user.id).order_by(Budget.created_at.desc()).all()
    transactions = db.query(Transaction).filter(
        Transaction.user_id == current_user.id,
        Transaction.type == "expense"
    ).all()
    result = []
    for budget in budgets:
        spent = sum(
            t.amount for t in transactions
            if (budget.category is None or t.category.lower() == budget.category.lower())
            and t.date.strftime("%Y-%m") == budget.month
        )
        utilization = round((spent / budget.amount * 100) if budget.amount else 0, 2)
        result.append({
            "id": budget.id,
            "name": budget.name,
            "category": budget.category,
            "amount": budget.amount,
            "month": budget.month,
            "spent": round(spent, 2),
            "remaining": round(budget.amount - spent, 2),
            "utilization": utilization,
            "status": "Exceeded" if utilization > 100 else "Warning" if utilization >= 80 else "Safe"
        })
    return result


@app.post("/budgets")
def create_budget(
    budget: BudgetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if budget.amount <= 0:
        raise HTTPException(status_code=400, detail="Budget amount must be greater than zero")
    item = Budget(
        user_id=current_user.id,
        name=budget.name,
        category=budget.category,
        amount=budget.amount,
        month=budget.month or datetime.utcnow().strftime("%Y-%m")
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"id": item.id, "name": item.name, "category": item.category, "amount": item.amount, "month": item.month}


@app.delete("/budgets/{budget_id}")
def delete_budget(
    budget_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    budget = db.query(Budget).filter(Budget.id == budget_id, Budget.user_id == current_user.id).first()
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    db.delete(budget)
    db.commit()
    return {"message": "Budget deleted"}


@app.get("/goals")
def get_goals(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    goals = db.query(FinancialGoal).filter(FinancialGoal.user_id == current_user.id).all()
    result = []
    for goal in goals:
        contributed = sum(c.amount for c in goal.contributions)
        progress = round(min(contributed / goal.target_amount * 100, 100), 2) if goal.target_amount else 0
        result.append({
            "id": goal.id,
            "name": goal.name,
            "target_amount": goal.target_amount,
            "target_date": goal.target_date,
            "contributed": round(contributed, 2),
            "remaining": round(max(goal.target_amount - contributed, 0), 2),
            "progress": progress
        })
    return result


@app.post("/goals")
def create_goal(
    goal: GoalCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if goal.target_amount <= 0:
        raise HTTPException(status_code=400, detail="Target amount must be greater than zero")
    item = FinancialGoal(
        user_id=current_user.id,
        name=goal.name,
        target_amount=goal.target_amount,
        target_date=goal.target_date
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"id": item.id, "name": item.name, "target_amount": item.target_amount, "target_date": item.target_date}


@app.post("/goals/{goal_id}/contributions")
def contribute_to_goal(
    goal_id: int,
    contribution: GoalContributionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    goal = db.query(FinancialGoal).filter(FinancialGoal.id == goal_id, FinancialGoal.user_id == current_user.id).first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    if contribution.amount <= 0:
        raise HTTPException(status_code=400, detail="Contribution must be greater than zero")
    item = GoalContribution(goal_id=goal.id, amount=contribution.amount)
    db.add(item)
    db.commit()
    return {"message": "Contribution recorded"}


@app.delete("/goals/{goal_id}")
def delete_goal(
    goal_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    goal = db.query(FinancialGoal).filter(FinancialGoal.id == goal_id, FinancialGoal.user_id == current_user.id).first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    db.delete(goal)
    db.commit()
    return {"message": "Goal deleted"}


@app.post("/feedback")
def submit_feedback(
    feedback: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if feedback.rating < 1 or feedback.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be from 1 to 5")
    item = Feedback(user_id=current_user.id, rating=feedback.rating, message=feedback.message.strip())
    if not item.message:
        raise HTTPException(status_code=400, detail="Feedback message is required")
    db.add(item)
    db.commit()
    return {"message": "Thank you for your feedback"}


@app.get("/reports/summary")
def report_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    income = sum(t.amount for t in transactions if t.type == "income")
    expenses = sum(t.amount for t in transactions if t.type == "expense")
    categories = {}
    for t in transactions:
        if t.type == "expense":
            categories[t.category] = categories.get(t.category, 0) + t.amount
    return {
        "income": round(income, 2),
        "expenses": round(expenses, 2),
        "savings": round(income - expenses, 2),
        "category_breakdown": categories,
        "transaction_count": len(transactions)
    }


@app.get("/reports/transactions.csv")
def export_transactions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).order_by(Transaction.date.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Type", "Category", "Description", "Amount"])
    for t in transactions:
        writer.writerow([t.date.isoformat(), t.type, t.category, t.description or "", t.amount])
    output.seek(0)
    return StreamingResponse(
        io.StringIO(output.getvalue()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=transactions.csv"}
    )


@app.get("/reports/summary.pdf")
def export_pdf_report(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    income = sum(t.amount for t in transactions if t.type == "income")
    expenses = sum(t.amount for t in transactions if t.type == "expense")
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=A4)
    pdf.setTitle("Financial Summary Report")
    pdf.drawString(50, 800, "Finova Financial Summary")
    pdf.drawString(50, 770, f"Account: {current_user.name}")
    pdf.drawString(50, 740, f"Location: {current_user.location}")
    pdf.drawString(50, 700, f"Total income: INR {income:,.2f}")
    pdf.drawString(50, 680, f"Total expenses: INR {expenses:,.2f}")
    pdf.drawString(50, 660, f"Net savings: INR {income - expenses:,.2f}")
    pdf.drawString(50, 620, f"Transactions: {len(transactions)}")
    pdf.drawString(50, 580, "Generated by Finova")
    pdf.save()
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=financial-summary.pdf"}
    )


@app.post("/assistant")
def assistant(
    request: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    question = (request.get("message") or "").lower()
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    income = sum(t.amount for t in transactions if t.type == "income")
    expenses = sum(t.amount for t in transactions if t.type == "expense")
    categories = {}
    for t in transactions:
        if t.type == "expense":
            categories[t.category] = categories.get(t.category, 0) + t.amount

    if "spend" in question or "expense" in question:
        answer = f"Your recorded expenses total ₹{expenses:,.0f}. " + (
            f"Your highest spending category is {max(categories, key=categories.get).title()} at ₹{max(categories.values()):,.0f}."
            if categories else "You have no expense records yet."
        )
    elif "save" in question:
        answer = f"Your current recorded savings are ₹{income - expenses:,.0f}. Automating ₹{max((income - expenses) * 0.2, 0):,.0f} per month is a practical starting point."
    elif "budget" in question or "exceed" in question:
        predicted = ml_engine.predict_next_month_spending(transactions)
        answer = f"The rolling estimate for upcoming spending is ₹{predicted:,.0f}, against your monthly income of ₹{current_user.monthly_income:,.0f}."
    elif "score" in question or "wellness" in question:
        score = ml_engine.calculate_stability_score(current_user, transactions)
        answer = f"Your financial wellness score is {score:.0f}/100."
    else:
        answer = "I can help with monthly spending, savings, budget overruns, the highest expense category, and your financial wellness score."
    return {"answer": answer}


@app.post("/demo/load")
def load_demo(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    existing = db.query(Transaction).filter(Transaction.user_id == current_user.id).count()
    if existing:
        raise HTTPException(status_code=400, detail="Demo data can only be loaded into an empty account")
    now = datetime.utcnow()
    income = Transaction(user_id=current_user.id, amount=50000, category="salary", type="income", description="Sample monthly salary", date=now)
    db.add(income)
    samples = [
        ("food", 8500, "Food and dining"), ("transport", 4200, "Transport"),
        ("shopping", 6500, "Shopping"), ("utilities", 4800, "Utilities"),
        ("entertainment", 3000, "Entertainment"), ("healthcare", 2300, "Healthcare"),
        ("education", 5500, "Education"), ("subscriptions", 2200, "Subscriptions")
    ]
    for index, (category, amount, description) in enumerate(samples):
        db.add(Transaction(
            user_id=current_user.id,
            amount=amount,
            category=category,
            type="expense",
            description=f"Sample {description}",
            date=now - timedelta(days=index * 3)
        ))
    db.commit()
    return {"message": "Sample data loaded"}


@app.delete("/account")
def delete_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db.query(Feedback).filter(Feedback.user_id == current_user.id).delete()
    db.query(Loan).filter(Loan.user_id == current_user.id).delete()
    db.query(Subscription).filter(Subscription.user_id == current_user.id).delete()
    db.query(ShoppingItem).filter(ShoppingItem.user_id == current_user.id).delete()
    db.query(Bill).filter(Bill.user_id == current_user.id).delete()
    db.query(SavingsChallenge).filter(SavingsChallenge.user_id == current_user.id).delete()
    db.query(CoachMessage).filter(CoachMessage.user_id == current_user.id).delete()
    groups = db.query(SharedExpenseGroup).filter(SharedExpenseGroup.owner_id == current_user.id).all()
    for group in groups:
        db.query(ManualSettlement).filter(ManualSettlement.group_id == group.id).delete()
        db.query(SharedExpense).filter(SharedExpense.group_id == group.id).delete()
        db.query(SharedGroupMember).filter(SharedGroupMember.group_id == group.id).delete()
        db.delete(group)
    db.query(Alert).filter(Alert.user_id == current_user.id).delete()
    db.query(Transaction).filter(Transaction.user_id == current_user.id).delete()
    db.query(Budget).filter(Budget.user_id == current_user.id).delete()
    goals = db.query(FinancialGoal).filter(FinancialGoal.user_id == current_user.id).all()
    for goal in goals:
        db.query(GoalContribution).filter(GoalContribution.goal_id == goal.id).delete()
        db.delete(goal)
    db.delete(current_user)
    db.commit()
    return {"message": "Account deleted"}

def generate_ai_suggestions(results: dict, samples: int, total_rows: int) -> list:
    """Generate intelligent AI suggestions based on model performance"""
    suggestions = []
    best_r2 = results['best_r2']
    linear_r2 = results['linear']['r2']
    best_rmse = results['linear']['rmse']
    
    # CNN comparison if available
    cnn_r2 = results.get('cnn', {}).get('r2', 0)
    if cnn_r2 > 0:
        best_rmse = min(best_rmse, results['cnn']['rmse'])
    
    # 1. Data Quality Analysis
    if samples < 1000:
        suggestions.append({
            "icon": "⚠️",
            "title": "Critical: Insufficient Training Data",
            "message": f"Only {samples:,} samples used. Model reliability is very low. Collect at least 5,000+ samples for basic predictions.",
            "color": "#fee2e2",
            "border": "#ef4444"
        })
    elif samples < 5000:
        suggestions.append({
            "icon": "📊",
            "title": "Data Volume: Below Recommended",
            "message": f"Training with {samples:,} samples. Increase to 10,000+ for production-grade accuracy (current: {best_r2*100:.1f}%, target: 85%+).",
            "color": "#fef3c7",
            "border": "#f59e0b"
        })
    
    # Additional warning for low R2
    if samples >= 5000 and best_r2 < 0.7:
        suggestions.append({
            "icon": "📈",
            "title": "Model Performance: Needs Improvement",
            "message": f"R² score of {best_r2*100:.1f}% indicates moderate accuracy. Add more diverse data or try feature engineering to reach 80%+ accuracy.",
            "color": "#fef3c7",
            "border": "#f59e0b"
        })
    
    # 2. Model Selection Insight
    if cnn_r2 > linear_r2 + 0.05:
        suggestions.append({
            "icon": "🧠",
            "title": "CNN Significantly Outperforms Linear Models",
            "message": f"CNN achieved {cnn_r2*100:.1f}% vs Linear's {linear_r2*100:.1f}%. Your data has complex patterns - use CNN for production deployment.",
            "color": "#fce7f3",
            "border": "#ec4899"
        })
    elif cnn_r2 > 0 and linear_r2 > cnn_r2:
        suggestions.append({
            "icon": "⚡",
            "title": "Linear Model is More Efficient",
            "message": f"Linear Regression ({linear_r2*100:.1f}%) outperforms CNN ({cnn_r2*100:.1f}%). Your data has linear relationships - stick with Linear for faster predictions.",
            "color": "#dbeafe",
            "border": "#3b82f6"
        })
    
    # 3. Practical Use Case Recommendation
    if best_r2 >= 0.85:
        use_case = "production-ready for critical financial decisions"
        reliability = "High reliability"
    elif best_r2 >= 0.70:
        use_case = "suitable for general expense forecasting and budgeting"
        reliability = "Good reliability"
    elif best_r2 >= 0.50:
        use_case = "useful for trend analysis, but verify predictions manually"
        reliability = "Moderate reliability"
    else:
        use_case = "only for rough estimates - not recommended for important decisions"
        reliability = "Low reliability"
    
    suggestions.append({
        "icon": "🎯",
        "title": f"Best Use Case: {reliability}",
        "message": f"This model is {use_case}. Expected error: ±₹{best_rmse:.0f} per prediction.",
        "color": "#dbeafe",
        "border": "#3b82f6"
    })
    
    return suggestions

def haversine_meters(lat1, lon1, lat2, lon2):
    radius = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@app.post("/advanced/nearby")
def nearby_places(request: dict, current_user: User = Depends(get_current_user)):
    try:
        lat = float(request["lat"])
        lon = float(request["lon"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Latitude and longitude are required")

    category = (request.get("category") or "restaurant").lower()
    radius = max(100, min(int(request.get("radius", 3000)), 10000))
    amenity_map = {
        "food": "restaurant|cafe|fast_food",
        "grocery": "convenience|supermarket",
        "pharmacy": "pharmacy",
        "hospital": "hospital|clinic",
        "fuel": "fuel",
        "transport": "bus_station|subway_entrance|train_station",
        "shopping": "department_store|mall"
    }
    amenity = amenity_map.get(category, "restaurant|cafe|fast_food")
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"{amenity}",i](around:{radius},{lat},{lon});
      way["amenity"~"{amenity}",i](around:{radius},{lat},{lon});
    );
    out center 40;
    """
    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query,
            timeout=20,
            headers={"User-Agent": "Finova-Financial-App/1.0"}
        )
        response.raise_for_status()
        elements = response.json().get("elements", [])
    except requests.RequestException as exc:
        return {
            "places": [],
            "source": "OpenStreetMap/Overpass",
            "verified_live_prices": False,
            "service_status": "unavailable",
            "service_message": "The nearby-place provider returned an error. Try again later or enter a location manually.",
            "error": str(exc)
        }

    places = []
    for item in elements:
        tags = item.get("tags", {})
        place_lat = item.get("lat") or item.get("center", {}).get("lat")
        place_lon = item.get("lon") or item.get("center", {}).get("lon")
        if place_lat is None or place_lon is None:
            continue
        places.append({
            "name": tags.get("name", "Unnamed place"),
            "category": tags.get("amenity", category),
            "address": ", ".join(filter(None, [
                tags.get("addr:housenumber"), tags.get("addr:street"),
                tags.get("addr:city"), tags.get("addr:postcode")
            ])) or "Address not provided",
            "lat": place_lat,
            "lon": place_lon,
            "distance_m": round(haversine_meters(lat, lon, place_lat, place_lon)),
            "rating": None,
            "opening_hours": tags.get("opening_hours"),
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "directions_url": f"https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route={lat}%2C{lon}%3B{place_lat}%2C{place_lon}",
            "price_source": "No verified menu prices returned by OpenStreetMap"
        })
    places.sort(key=lambda place: place["distance_m"])
    return {"places": places, "source": "OpenStreetMap/Overpass", "verified_live_prices": False}


@app.post("/advanced/geocode")
def geocode_location(request: dict):
    query = (request.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Location query is required")
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 5},
            headers={"User-Agent": "Finova-Financial-App/1.0"},
            timeout=15
        )
        response.raise_for_status()
        return {"results": [
            {"name": item["display_name"], "lat": float(item["lat"]), "lon": float(item["lon"])}
            for item in response.json()
        ], "source": "OpenStreetMap/Nominatim"}
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Location service unavailable: {exc}")


@app.post("/advanced/grocery")
def grocery_assistant(request: dict, current_user: User = Depends(get_current_user)):
    budget = float(request.get("budget", 0))
    if budget <= 0:
        raise HTTPException(status_code=400, detail="Budget must be greater than zero")
    essential = [
        ("Rice", 5, 70), ("Dal", 1, 140), ("Vegetables", 2, 80),
        ("Milk", 1, 60), ("Eggs", 6, 72), ("Cooking oil", 1, 160),
        ("Bread", 1, 45), ("Onion", 1, 40)
    ]
    items = []
    total = 0
    for name, quantity, estimated in essential:
        if total + estimated <= budget:
            items.append({"name": name, "quantity": f"{quantity} unit(s)", "estimated_price": estimated, "price_status": "estimated"})
            total += estimated
    return {
        "items": items,
        "estimated_total": total,
        "remaining_budget": round(budget - total, 2),
        "estimate_notice": "Prices are local planning estimates, not verified store prices.",
        "manual_confirmation_required": True
    }


@app.post("/advanced/purchase")
def purchase_decision(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    price = float(request.get("price", 0))
    if price < 0:
        raise HTTPException(status_code=400, detail="Price cannot be negative")
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    income = sum(t.amount for t in transactions if t.type == "income")
    expenses = sum(t.amount for t in transactions if t.type == "expense")
    balance = income - expenses
    goal_approach = sum(g.target_amount - sum(c.amount for c in g.contributions) for g in db.query(FinancialGoal).filter(FinancialGoal.user_id == current_user.id).all())
    remaining = balance - price
    ratio = price / current_user.monthly_income if current_user.monthly_income else 1
    if remaining >= 0 and ratio <= 0.1:
        decision, score = "Affordable", 90
    elif remaining >= 0 and ratio <= 0.25:
        decision, score = "Affordable with caution", 65
    else:
        decision, score = "Not recommended", 25
    return {
        "decision": decision,
        "affordability_score": score,
        "available_balance": round(balance, 2),
        "remaining_balance": round(remaining, 2),
        "effect_on_savings": round(-price, 2),
        "goal_funding_at_risk": round(min(goal_approach, max(price, 0)), 2),
        "explanation": "Based on your recorded balance, monthly income, current expenses and savings goals."
    }


@app.post("/advanced/allowance")
def daily_allowance(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    remaining_days = max(int(request.get("remaining_days", 30)), 1)
    available = request.get("available")
    if available is None:
        transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
        available = sum(t.amount for t in transactions if t.type == "income") - sum(t.amount for t in transactions if t.type == "expense")
    available = max(float(available), 0)
    daily = available / remaining_days
    return {
        "available_money": round(available, 2),
        "remaining_days": remaining_days,
        "daily_allowance": round(daily, 2),
        "weekly_allowance": round(daily * 7, 2),
        "exhaustion_risk": "HIGH" if daily < 200 else "MEDIUM" if daily < 500 else "LOW",
        "recommendation": "Protect essential food, healthcare and housing first."
    }


@app.post("/advanced/emergency")
def emergency_budget(request: dict, current_user: User = Depends(get_current_user)):
    available = max(float(request.get("available", 0)), 0)
    days = max(int(request.get("remaining_days", 1)), 1)
    daily = available / days
    essential = {
        "food": round(daily * 0.45, 2),
        "transport": round(daily * 0.20, 2),
        "utilities": round(daily * 0.20, 2),
        "healthcare_reserve": round(daily * 0.10, 2),
        "contingency": round(daily * 0.05, 2)
    }
    return {
        "available": available,
        "remaining_days": days,
        "daily_emergency_allowance": round(daily, 2),
        "essential_allocation": essential,
        "warning": "Do not reduce essential food, healthcare or housing. The plan prioritizes essentials."
    }


@app.post("/advanced/simulate")
def financial_simulation(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    actual_income = sum(t.amount for t in transactions if t.type == "income") or current_user.monthly_income
    actual_expenses = sum(t.amount for t in transactions if t.type == "expense")
    income_change = float(request.get("income_change", 0))
    expense_change = float(request.get("expense_change", 0))
    extra_savings = float(request.get("extra_savings", 0))
    unexpected = float(request.get("unexpected_expense", 0))
    purchase = float(request.get("purchase", 0))
    emi = float(request.get("new_emi", 0))
    monthly_income = actual_income + income_change
    monthly_expense = actual_expenses + expense_change + unexpected / 30 + emi
    monthly_net = monthly_income - monthly_expense - purchase / 30 - extra_savings
    horizon = {}
    for label, months in (("30_days", 1), ("3_months", 3), ("6_months", 6), ("12_months", 12)):
        horizon[label] = round(monthly_net * months, 2)
    return {
        "simulated": True,
        "base_monthly_income": round(actual_income, 2),
        "base_monthly_expenses": round(actual_expenses, 2),
        "projected_monthly_balance": round(monthly_net, 2),
        "projected_outcomes": horizon,
        "notice": "These are scenario calculations, not guaranteed forecasts."
    }


@app.post("/advanced/emergency-fund")
def emergency_fund(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    monthly_expenses = sum(t.amount for t in transactions if t.type == "expense") or current_user.monthly_income * 0.7
    existing = max(float(request.get("existing", 0)), 0)
    target_months = int(request.get("months", 6))
    target_months = min(max(target_months, 3), 12)
    target = monthly_expenses * target_months
    remaining = max(target - existing, 0)
    contribution = remaining / 12
    return {
        "monthly_expense_baseline": round(monthly_expenses, 2),
        "three_month_target": round(monthly_expenses * 3, 2),
        "six_month_target": round(monthly_expenses * 6, 2),
        "twelve_month_target": round(monthly_expenses * 12, 2),
        "selected_target": round(target, 2),
        "existing_fund": round(existing, 2),
        "remaining": round(remaining, 2),
        "suggested_monthly_contribution": round(contribution, 2),
        "readiness_percent": round(existing / target * 100, 2) if target else 100
    }


@app.get("/advanced/loans")
def get_loans(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    loans = db.query(Loan).filter(Loan.user_id == current_user.id).all()
    income = current_user.monthly_income
    total_emi = sum(l.emi for l in loans)
    ratio = total_emi / income if income else 0
    return {
        "loans": [{
            "id": l.id, "name": l.name, "principal": l.principal, "outstanding": l.outstanding,
            "annual_interest": l.annual_interest, "emi": l.emi,
            "next_payment": l.next_payment.isoformat() if l.next_payment else None
        } for l in loans],
        "total_monthly_emi": round(total_emi, 2),
        "debt_to_income_ratio": round(ratio * 100, 2),
        "strategy": "Avalanche prioritizes the highest interest rate; snowball prioritizes the smallest outstanding balance."
    }


@app.post("/advanced/loans")
def create_loan(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        loan = Loan(
            user_id=current_user.id,
            name=request["name"],
            principal=float(request["principal"]),
            outstanding=float(request.get("outstanding", request["principal"])),
            annual_interest=float(request.get("annual_interest", 0)),
            emi=float(request["emi"]),
            next_payment=datetime.fromisoformat(request["next_payment"].replace("Z", "+00:00")) if request.get("next_payment") else None
        )
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Name, principal and EMI are required")
    db.add(loan)
    db.commit()
    db.refresh(loan)
    return {"id": loan.id, "message": "Loan added"}


@app.get("/advanced/subscriptions")
def get_subscriptions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    subscriptions = db.query(Subscription).filter(Subscription.user_id == current_user.id, Subscription.active == True).all()
    monthly = sum(s.amount for s in subscriptions)
    return {
        "subscriptions": [{
            "id": s.id, "name": s.name, "amount": s.amount,
            "next_payment": s.next_payment.isoformat() if s.next_payment else None
        } for s in subscriptions],
        "monthly_cost": round(monthly, 2),
        "annual_cost": round(monthly * 12, 2),
        "notice": "No cancellation is performed automatically."
    }


@app.post("/advanced/subscriptions")
def create_subscription(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        subscription = Subscription(
            user_id=current_user.id,
            name=request["name"],
            amount=float(request["amount"]),
            next_payment=datetime.fromisoformat(request["next_payment"].replace("Z", "+00:00")) if request.get("next_payment") else None
        )
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Name and amount are required")
    db.add(subscription)
    db.commit()
    db.refresh(subscription)
    return {"id": subscription.id, "message": "Subscription added"}


@app.get("/advanced/anomalies")
def spending_anomalies(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id, Transaction.type == "expense").all()
    if not transactions:
        return {"anomalies": [], "message": "No expense history available"}
    amounts = [t.amount for t in transactions]
    mean = sum(amounts) / len(amounts)
    variance = sum((amount - mean) ** 2 for amount in amounts) / len(amounts)
    std = math.sqrt(variance)
    anomalies = []
    for transaction in transactions:
        z_score = (transaction.amount - mean) / std if std else 0
        if z_score > 2:
            anomalies.append({
                "transaction_id": transaction.id,
                "category": transaction.category,
                "amount": transaction.amount,
                "date": transaction.date.isoformat(),
                "reason": "Amount is statistically unusual relative to this account.",
                "requires_user_review": True
            })
    return {"anomalies": anomalies, "method": "Statistical Z-score", "fraud_claim": False}


@app.get("/advanced/calendar")
def financial_calendar(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    loans = db.query(Loan).filter(Loan.user_id == current_user.id).all()
    subscriptions = db.query(Subscription).filter(Subscription.user_id == current_user.id, Subscription.active == True).all()
    items = []
    for loan in loans:
        if loan.next_payment:
            items.append({"date": loan.next_payment.isoformat(), "title": f"{loan.name} EMI", "amount": loan.emi, "type": "emi"})
    for subscription in subscriptions:
        if subscription.next_payment:
            items.append({"date": subscription.next_payment.isoformat(), "title": f"{subscription.name} subscription", "amount": subscription.amount, "type": "subscription"})
    items.sort(key=lambda item: item["date"])
    return {"events": items}


@app.post("/advanced/coach")
def location_financial_coach(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    message = (request.get("message") or "").lower()
    budget = float(request.get("budget", 0) or 0)
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    expenses = sum(t.amount for t in transactions if t.type == "expense")
    balance = sum(t.amount for t in transactions if t.type == "income") - expenses
    if "eat" in message or "food" in message:
        answer = "Check the nearby food options before buying. Use the map and filters; menu prices are estimates unless confirmed by the place."
    elif "grocery" in message:
        answer = "Use a grocery plan with essential items. Confirm in-store prices before checkout."
    elif "travel" in message or "transport" in message:
        answer = "Compare public transport options; fares and schedules should be confirmed with the current provider."
    elif "afford" in message or "purchase" in message:
        answer = f"Use the purchase checker. Your recorded available balance is ₹{balance:,.0f}; spending ₹{budget:,.0f} would leave ₹{balance-budget:,.0f}."
    elif "emergency" in message:
        answer = "Prioritize essential food, healthcare and housing. Calculate a per-day allowance for the exact remaining period."
    else:
        answer = "I can help with food, groceries, travel, purchases, emergency budgeting and daily allowances."
    return {"answer": answer, "recorded_balance": round(balance, 2), "budget": budget}


@app.post("/finmate/survival")
def money_survival(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    income = sum(t.amount for t in transactions if t.type == "income")
    expenses = sum(t.amount for t in transactions if t.type == "expense")
    balance = float(request.get("available_balance", income - expenses))
    balance = max(balance, 0)
    days = max(int(request.get("days_until_income", 30)), 1)
    expenses_30 = [t.amount for t in transactions if t.type == "expense" and t.date >= datetime.utcnow() - timedelta(days=30)]
    daily_spending = sum(expenses_30) / 30 if expenses_30 else expenses / 30
    upcoming = db.query(Bill).filter(Bill.user_id == current_user.id, Bill.paid == False).all()
    upcoming_bills = sum(b.amount for b in upcoming if b.due_date <= datetime.utcnow() + timedelta(days=days))
    reserve = current_user.emergency_reserve or 0
    usable = max(balance - upcoming_bills - reserve, 0)
    predicted_days = int(usable / daily_spending) if daily_spending > 0 else days
    predicted_days = min(predicted_days, days)
    safe_daily = usable / days
    shortage = datetime.utcnow() + timedelta(days=predicted_days)
    risk = "RED" if predicted_days < days * 0.5 else "YELLOW" if predicted_days < days else "GREEN"
    return {
        "available_balance": round(balance, 2),
        "upcoming_bills": round(upcoming_bills, 2),
        "emergency_reserve": round(reserve, 2),
        "average_daily_spending": round(daily_spending, 2),
        "expected_remaining_days": predicted_days,
        "predicted_shortage_date": shortage.date().isoformat(),
        "safe_daily_spending": round(safe_daily, 2),
        "risk": risk,
        "recommendations": [
            "Reduce discretionary spending if predicted days are below the next-income date.",
            "Prioritize essential food, housing, healthcare and upcoming bills.",
            "This estimate improves as more transactions are recorded."
        ],
        "method": "Recorded balance, 30-day historical average and upcoming bills"
    }


@app.get("/finmate/risk")
def financial_risk(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    income = sum(t.amount for t in transactions if t.type == "income")
    expenses = sum(t.amount for t in transactions if t.type == "expense")
    balance = income - expenses
    bills = db.query(Bill).filter(Bill.user_id == current_user.id, Bill.paid == False).all()
    overdue = sum(b.amount for b in bills if b.due_date < datetime.utcnow())
    upcoming = sum(b.amount for b in bills if b.due_date >= datetime.utcnow())
    warnings = []
    if balance < 0 or expenses > income:
        warnings.append({"reason": "Recorded expenses exceed recorded income.", "data": f"Balance: INR {balance:,.0f}", "consequence": "A shortage or missed payment may occur.", "action": "Review and reduce non-essential spending."})
    if overdue:
        warnings.append({"reason": "Overdue payments are recorded.", "data": f"INR {overdue:,.0f} overdue", "consequence": "Late fees or service interruptions may occur.", "action": "Prioritize overdue payments."})
    if upcoming > max(balance, 0):
        warnings.append({"reason": "Upcoming bills exceed the available balance.", "data": f"Bills: INR {upcoming:,.0f}", "consequence": "Future obligations may be difficult to cover.", "action": "Review the payment calendar and adjust spending."})
    level = "RED" if len(warnings) >= 2 or expenses > income else "YELLOW" if warnings else "GREEN"
    return {"level": level, "balance": round(balance, 2), "upcoming_bills": round(upcoming, 2), "warnings": warnings}


@app.post("/finmate/bills")
def add_bill(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        name = request["name"].strip()
        amount = float(request["amount"])
        due_date = datetime.fromisoformat(request["due_date"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Name, amount and due date are required")
    if not name or amount <= 0:
        raise HTTPException(status_code=400, detail="Enter a valid name and positive amount")
    bill = Bill(
        user_id=current_user.id,
        name=name,
        amount=amount,
        due_date=due_date,
        recurring=bool(request.get("recurring", False))
    )
    db.add(bill)
    db.commit()
    db.refresh(bill)
    return {"id": bill.id, "message": "Bill added"}


@app.get("/finmate/bills")
def get_bills(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bills = db.query(Bill).filter(Bill.user_id == current_user.id).order_by(Bill.due_date.asc()).all()
    upcoming_total = sum(b.amount for b in bills if not b.paid)
    return {
        "bills": [{
            "id": b.id, "name": b.name, "amount": b.amount,
            "due_date": b.due_date.isoformat(), "recurring": b.recurring,
            "paid": b.paid,
            "status": "paid" if b.paid else "overdue" if b.due_date < datetime.utcnow() else "upcoming"
        } for b in bills],
        "upcoming_total": round(upcoming_total, 2)
    }


@app.put("/finmate/bills/{bill_id}/paid")
def mark_bill_paid(bill_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bill = db.query(Bill).filter(Bill.id == bill_id, Bill.user_id == current_user.id).first()
    if not bill:
        raise HTTPException(status_code=404, detail="Bill not found")
    bill.paid = True
    db.commit()
    return {"message": "Bill marked as paid"}


@app.delete("/finmate/bills/{bill_id}")
def delete_bill(bill_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bill = db.query(Bill).filter(Bill.id == bill_id, Bill.user_id == current_user.id).first()
    if not bill:
        raise HTTPException(status_code=404, detail="Bill not found")
    db.delete(bill)
    db.commit()
    return {"message": "Bill deleted"}


@app.post("/finmate/groups")
def create_shared_group(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    name = (request.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Group name is required")
    group = SharedExpenseGroup(owner_id=current_user.id, name=name)
    db.add(group)
    db.commit()
    db.refresh(group)
    return {"id": group.id, "name": group.name}


@app.get("/finmate/groups")
def get_shared_groups(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    groups = db.query(SharedExpenseGroup).filter(SharedExpenseGroup.owner_id == current_user.id).all()
    result = []
    for group in groups:
        members = db.query(SharedGroupMember).filter(SharedGroupMember.group_id == group.id).all()
        expenses = db.query(SharedExpense).filter(SharedExpense.group_id == group.id).all()
        result.append({"id": group.id, "name": group.name, "members": [m.name for m in members],
                       "expense_count": len(expenses), "total": round(sum(e.amount for e in expenses), 2)})
    return {"groups": result}


@app.post("/finmate/groups/{group_id}/members")
def add_group_member(group_id: int, request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    group = db.query(SharedExpenseGroup).filter(SharedExpenseGroup.id == group_id, SharedExpenseGroup.owner_id == current_user.id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    name = (request.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Member name is required")
    existing = db.query(SharedGroupMember).filter(SharedGroupMember.group_id == group_id, SharedGroupMember.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Member already exists")
    db.add(SharedGroupMember(group_id=group_id, name=name))
    db.commit()
    return {"message": "Member added"}


@app.post("/finmate/groups/{group_id}/expenses")
def add_shared_expense(group_id: int, request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    group = db.query(SharedExpenseGroup).filter(SharedExpenseGroup.id == group_id, SharedExpenseGroup.owner_id == current_user.id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    try:
        amount = float(request["amount"])
        payer = request["payer"].strip()
        description = request["description"].strip()
        participants = [p.strip() for p in request["participants"] if p.strip()]
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Amount, payer, description and participants are required")
    if amount <= 0 or payer not in participants or not participants:
        raise HTTPException(status_code=400, detail="Payer must participate and amount must be positive")
    db.add(SharedExpense(
        group_id=group_id,
        payer_name=payer,
        description=description,
        amount=amount,
        participant_names="|".join(participants),
        split_mode=request.get("split_mode", "equal")
    ))
    db.commit()
    return {"message": "Shared expense recorded"}


@app.get("/finmate/groups/{group_id}/balances")
def shared_balances(group_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    group = db.query(SharedExpenseGroup).filter(SharedExpenseGroup.id == group_id, SharedExpenseGroup.owner_id == current_user.id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    balances = {member.name: 0.0 for member in db.query(SharedGroupMember).filter(SharedGroupMember.group_id == group_id).all()}
    expenses = db.query(SharedExpense).filter(SharedExpense.group_id == group_id).all()
    for expense in expenses:
        participants = [p for p in expense.participant_names.split("|") if p]
        share = expense.amount / len(participants) if participants else 0
        balances[expense.payer_name] = balances.get(expense.payer_name, 0) + expense.amount
        for participant in participants:
            balances[participant] = balances.get(participant, 0) - share
    settlements = []
    debtors = sorted([(name, -balance) for name, balance in balances.items() if balance < -0.01])
    creditors = sorted([(name, balance) for name, balance in balances.items() if balance > 0.01], key=lambda x: x[1], reverse=True)
    i = j = 0
    while i < len(debtors) and j < len(creditors):
        debtor, owed = debtors[i]
        creditor, credit = creditors[j]
        amount = round(min(owed, credit), 2)
        if amount:
            settlements.append({"from": debtor, "to": creditor, "amount": amount})
        owed -= amount
        credit -= amount
        debtors[i] = (debtor, owed)
        creditors[j] = (creditor, credit)
        if owed <= 0.01:
            i += 1
        if credit <= 0.01:
            j += 1
    for settlement in db.query(ManualSettlement).filter(ManualSettlement.group_id == group_id, ManualSettlement.completed == False).all():
        settlements.append({"from": settlement.payer_name, "to": settlement.recipient_name, "amount": settlement.amount, "settlement_id": settlement.id})
    return {"balances": {k: round(v, 2) for k, v in balances.items()}, "suggested_settlements": settlements}


@app.post("/finmate/groups/{group_id}/settlements")
def record_settlement(group_id: int, request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    group = db.query(SharedExpenseGroup).filter(SharedExpenseGroup.id == group_id, SharedExpenseGroup.owner_id == current_user.id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    try:
        payer = request["from"].strip()
        recipient = request["to"].strip()
        amount = float(request["amount"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="From, to and amount are required")
    if payer == recipient or amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid settlement")
    duplicate = db.query(ManualSettlement).filter(
        ManualSettlement.group_id == group_id,
        ManualSettlement.payer_name == payer,
        ManualSettlement.recipient_name == recipient,
        ManualSettlement.amount == amount,
        ManualSettlement.completed == False
    ).first()
    if duplicate:
        raise HTTPException(status_code=400, detail="This settlement is already pending")
    db.add(ManualSettlement(group_id=group_id, payer_name=payer, recipient_name=recipient, amount=amount))
    db.commit()
    return {"message": "Settlement recorded manually"}


@app.post("/finmate/challenges")
def create_challenge(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        name = request["name"].strip()
        target = float(request["target_amount"])
        cadence = request["cadence"].lower()
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Name, target amount and cadence are required")
    if not name or target <= 0 or cadence not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="Use a valid name, positive target and daily, weekly or monthly cadence")
    challenge = SavingsChallenge(user_id=current_user.id, name=name, target_amount=target, cadence=cadence)
    db.add(challenge)
    db.commit()
    db.refresh(challenge)
    return {"id": challenge.id, "message": "Savings challenge started"}


@app.get("/finmate/challenges")
def get_challenges(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    challenges = db.query(SavingsChallenge).filter(SavingsChallenge.user_id == current_user.id).all()
    return {"challenges": [{
        "id": c.id, "name": c.name, "target_amount": c.target_amount,
        "current_amount": c.current_amount,
        "cadence": c.cadence, "status": c.status,
        "progress": round(min(c.current_amount / c.target_amount * 100, 100), 2) if c.target_amount else 0
    } for c in challenges]}


@app.post("/finmate/challenges/{challenge_id}/progress")
def update_challenge_progress(challenge_id: int, request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    challenge = db.query(SavingsChallenge).filter(SavingsChallenge.id == challenge_id, SavingsChallenge.user_id == current_user.id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    try:
        amount = float(request["amount"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Savings amount is required")
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Savings amount must be positive")
    challenge.current_amount = min(challenge.current_amount + amount, challenge.target_amount)
    if challenge.current_amount >= challenge.target_amount:
        challenge.status = "completed"
    db.commit()
    return {"message": "Savings progress recorded", "status": challenge.status}


@app.put("/finmate/challenges/{challenge_id}/status")
def update_challenge_status(challenge_id: int, request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    challenge = db.query(SavingsChallenge).filter(SavingsChallenge.id == challenge_id, SavingsChallenge.user_id == current_user.id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    status = request.get("status")
    if status not in ("active", "paused", "cancelled"):
        raise HTTPException(status_code=400, detail="Invalid challenge status")
    challenge.status = status
    db.commit()
    return {"message": f"Challenge {status}"}


@app.post("/finmate/daily-limit")
def calculate_daily_limit(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    balance = float(request.get("available_balance", sum(t.amount for t in transactions if t.type == "income") - sum(t.amount for t in transactions if t.type == "expense")))
    days = max(int(request.get("days_remaining", 30)), 1)
    upcoming = sum(b.amount for b in db.query(Bill).filter(Bill.user_id == current_user.id, Bill.paid == False).all())
    reserve = current_user.emergency_reserve or 0
    usable = max(balance - upcoming - reserve, 0)
    daily = usable / days
    weekly = daily * 7
    return {
        "available_balance": round(balance, 2),
        "upcoming_obligations": round(upcoming, 2),
        "emergency_reserve": round(reserve, 2),
        "days_remaining": days,
        "daily_safe_limit": round(daily, 2),
        "weekly_safe_limit": round(weekly, 2),
        "explanation": "Limit = (balance - upcoming bills - emergency reserve) divided by remaining days."
    }


@app.get("/finmate/spending-pattern")
def spending_pattern(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    now = datetime.utcnow()
    current = db.query(Transaction).filter(
        Transaction.user_id == current_user.id,
        Transaction.type == "expense",
        Transaction.date >= now - timedelta(days=30)
    ).all()
    previous = db.query(Transaction).filter(
        Transaction.user_id == current_user.id,
        Transaction.type == "expense",
        Transaction.date >= now - timedelta(days=60),
        Transaction.date < now - timedelta(days=30)
    ).all()
    current_categories = {}
    previous_categories = {}
    for transaction in current:
        current_categories[transaction.category] = current_categories.get(transaction.category, 0) + transaction.amount
    for transaction in previous:
        previous_categories[transaction.category] = previous_categories.get(transaction.category, 0) + transaction.amount
    trends = []
    for category, amount in current_categories.items():
        prior = previous_categories.get(category, 0)
        change = ((amount - prior) / prior * 100) if prior else None
        if change is not None:
            trends.append({"category": category, "current": round(amount, 2), "previous": round(prior, 2),
                           "change_percent": round(change, 2)})
    amounts = [t.amount for t in current]
    mean = sum(amounts) / len(amounts) if amounts else 0
    anomaly = [{"id": t.id, "category": t.category, "amount": t.amount, "reason": "Unusually large for this account"}
               for t in current if mean and t.amount > mean * 2]
    return {"current_categories": current_categories, "previous_categories": previous_categories,
            "trends": sorted(trends, key=lambda x: abs(x["change_percent"]), reverse=True),
            "unusual_transactions": anomaly,
            "method": "30-day category totals compared with the preceding 30 days"}


@app.get("/finmate/coach/history")
def coach_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    messages = db.query(CoachMessage).filter(CoachMessage.user_id == current_user.id).order_by(CoachMessage.created_at.asc()).limit(100).all()
    return {"messages": [{"role": m.role, "message": m.message, "created_at": m.created_at.isoformat()} for m in messages]}


@app.post("/finmate/coach")
def finmate_coach(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    question = (request.get("message") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message is required")
    transactions = db.query(Transaction).filter(Transaction.user_id == current_user.id).all()
    income = sum(t.amount for t in transactions if t.type == "income")
    expenses = sum(t.amount for t in transactions if t.type == "expense")
    balance = income - expenses
    days = 30
    daily = balance / days if balance > 0 else 0
    lower = question.lower()
    if "spend" in lower and ("today" in lower or "300" in lower):
        answer = f"Based on your recorded balance of INR {balance:,.0f}, a basic safe daily estimate is INR {daily:,.0f}. Upcoming bills and your emergency reserve should be deducted first."
    elif "save" in lower:
        answer = f"Your recorded savings balance is INR {balance:,.0f}. A practical monthly target may be 20% of recorded income, but this should account for upcoming bills and goals."
    elif "expense" in lower and ("increase" in lower or "why" in lower):
        answer = "Review the spending-pattern page to compare category totals for the current and previous 30-day periods."
    elif "emergency" in lower:
        answer = f"Your emergency reserve setting is INR {current_user.emergency_reserve or 0:,.0f}. A general planning reference is 3-6 months of essential expenses, adapted to your needs."
    elif "salary" in lower or "money last" in lower:
        answer = f"Your recorded balance is INR {balance:,.0f}. Use the Money Survival Predictor with your next-income date and upcoming bills for a clear estimate."
    else:
        answer = f"I can use your recorded balance of INR {balance:,.0f}, income and expense history to explain spending and savings decisions. No external AI claim is made."
    db.add(CoachMessage(user_id=current_user.id, role="user", message=question))
    db.add(CoachMessage(user_id=current_user.id, role="assistant", message=answer))
    db.commit()
    return {"answer": answer, "recorded_balance": round(balance, 2), "follow_up": "Would you like me to calculate your safe daily limit or review upcoming bills?"}


@app.post("/advanced/travel")
def affordable_travel(request: dict):
    origin = (request.get("origin") or "").strip()
    destination = (request.get("destination") or "").strip()
    budget = max(float(request.get("budget", 0)), 0)
    if not origin or not destination:
        raise HTTPException(status_code=400, detail="Origin and destination are required")
    headers = {"User-Agent": "Finova-Financial-App/1.0"}
    try:
        geocoded = []
        for place in (origin, destination):
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": place, "format": "json", "limit": 1},
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            results = response.json()
            if not results:
                raise HTTPException(status_code=404, detail=f"Could not geocode {place}")
            geocoded.append((float(results[0]["lon"]), float(results[0]["lat"])))
        start, end = geocoded
        route_url = f"http://router.project-osrm.org/route/v1/driving/{start[0]},{start[1]};{end[0]},{end[1]}"
        route = requests.get(route_url, params={"overview": "false"}, headers=headers, timeout=20)
        route.raise_for_status()
        route_data = route.json().get("routes", [{}])[0]
        distance_km = round(route_data.get("distance", 0) / 1000, 2)
        drive_minutes = round(route_data.get("duration", 0) / 60)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Route service unavailable: {exc}")

    walking_hours = round(distance_km / 4.5, 1)
    bus_estimate = round(max(distance_km * 1.2, 20), 2)
    rail_estimate = round(max(distance_km * 0.8, 15), 2)
    options = [
        {"type": "bus", "estimated_fare": bus_estimate, "distance_km": distance_km,
         "travel_time_minutes": drive_minutes + 20, "fare_status": "estimated", "affordable": bus_estimate <= budget},
        {"type": "rail_or_metro", "estimated_fare": rail_estimate, "distance_km": distance_km,
         "travel_time_minutes": max(drive_minutes - 10, 15), "fare_status": "estimated", "affordable": rail_estimate <= budget},
        {"type": "walking", "estimated_fare": 0, "distance_km": distance_km,
         "travel_time_hours": walking_hours, "fare_status": "verified/no fare", "affordable": 0 <= budget}
    ]
    return {
        "origin": origin,
        "destination": destination,
        "budget": budget,
        "remaining_budget_by_option": {
            option["type"]: round(budget - option["estimated_fare"], 2) for option in options
        },
        "options": options,
        "source": "OpenStreetMap/Nominatim + OSRM",
        "schedule_notice": "Live schedules and fares are not guaranteed; confirm with the transport provider.",
        "map_url": f"https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route={start[1]}%2C{start[0]}%3B{end[1]}%2C{end[0]}"
    }


@app.post("/advanced/shopping-list")
def save_shopping_item(request: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    name = (request.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Item name is required")
    item = ShoppingItem(
        user_id=current_user.id,
        name=name,
        quantity=request.get("quantity"),
        estimated_price=max(float(request.get("estimated_price", 0)), 0)
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"id": item.id, "message": "Shopping item saved"}


@app.get("/advanced/shopping-list")
def get_shopping_list(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    items = db.query(ShoppingItem).filter(ShoppingItem.user_id == current_user.id).order_by(ShoppingItem.created_at.desc()).all()
    return {"items": [{
        "id": item.id, "name": item.name, "quantity": item.quantity,
        "estimated_price": item.estimated_price, "purchased": item.purchased
    } for item in items]}


@app.post("/advanced/price-compare")
def compare_prices(request: dict):
    offers = request.get("offers")
    if not isinstance(offers, list) or not offers:
        raise HTTPException(status_code=400, detail="Provide at least one manual store offer")
    normalized = []
    for offer in offers:
        try:
            price = float(offer["price"])
            store = str(offer["store"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if price < 0 or not store:
            continue
        normalized.append({"store": store, "price": price, "product": offer.get("product", "Product")})
    if not normalized:
        raise HTTPException(status_code=400, detail="No valid offers provided")
    best = min(normalized, key=lambda item: item["price"])
    original = max(normalized, key=lambda item: item["price"])
    return {
        "offers": normalized,
        "best_value": best,
        "potential_savings": round(original["price"] - best["price"], 2),
        "source_status": "Manual user-entered prices only; no live offer data assumed."
    }


@app.post("/advanced/deals")
def find_deals(request: dict):
    return {
        "deals": [],
        "budget": max(float(request.get("budget", 0)), 0),
        "source_status": "No supported deal API is configured.",
        "notice": "No offers are invented or presented as verified.",
        "required_configuration": "Configure GOOGLE_PLACES_API_KEY or a supported merchant/deal API for verified offers."
    }


@app.post("/spending-predict")
def spending_predict(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    budget = request.get("budget", 100)
    food_item = request.get("food_item", "thali")
    location = request.get("location") or current_user.location or "India"
    return predict_spending(budget, food_item, location)

@app.post("/train-model")
async def train_model(file: UploadFile = File(...), samples: int = Form(...), train_cnn: str = Form("false")):
    from src.ml.advanced_ml_engine import AdvancedFinancialMLEngine
    import time
    
    train_cnn_bool = train_cnn.lower() == "true"
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    if samples < 10 or samples > 250000:
        raise HTTPException(status_code=400, detail="Samples must be between 10 and 250000")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            content = file.file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        df = pd.read_csv(tmp_path)
        total_rows = len(df)
        
        if samples > total_rows:
            samples = total_rows
        
        ml_engine_adv = AdvancedFinancialMLEngine()
        results = ml_engine_adv.train_all_models(csv_path=tmp_path, n_samples=samples)
        
        train_size = int(samples * 0.8)
        test_size = samples - train_size
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        
        response = {
            "total_rows": total_rows,
            "samples_used": samples,
            "train_size": train_size,
            "test_size": test_size,
            "linear": results['linear'],
            "ridge": results['ridge'],
            "lasso": results['lasso'],
            "best_model": best_model[0].title(),
            "best_r2": best_model[1]['r2']
        }
        
        if train_cnn_bool:
            try:
                from src.ml.hybrid_cnn_model import HybridCNNModel
                start_time = time.time()
                cnn_model = HybridCNNModel()
                cnn_results = cnn_model.train(csv_path=tmp_path, n_samples=samples, epochs=50)
                training_time = time.time() - start_time
                response['cnn'] = {
                    'mae': cnn_results['mae'],
                    'rmse': cnn_results['rmse'],
                    'r2': cnn_results['r2'],
                    'training_time': f"{int(training_time // 60)}m {int(training_time % 60)}s"
                }
            except Exception as e:
                response['cnn_error'] = str(e)
        
        # Generate AI suggestions
        print(f"Generating AI suggestions for {samples} samples with R2={response['best_r2']}")
        response['ai_suggestions'] = generate_ai_suggestions(response, samples, total_rows)
        print(f"Generated {len(response['ai_suggestions'])} suggestions")
        
        os.unlink(tmp_path)
        return response
    except Exception as e:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))
