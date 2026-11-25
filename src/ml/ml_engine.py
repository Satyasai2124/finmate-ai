import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.api.config import Config

class FinancialMLEngine:
    def __init__(self):
        self.model = LinearRegression()
    
    def analyze_spending_pattern(self, transactions):
        if not transactions:
            return {}
        
        df = pd.DataFrame([{
            'amount': t.amount,
            'category': t.category,
            'type': t.type,
            'date': t.date
        } for t in transactions])
        
        category_spending = df[df['type'] == 'expense'].groupby('category')['amount'].sum().to_dict()
        return category_spending
    
    def predict_next_month_spending(self, transactions):
        if len(transactions) < Config.MIN_TRANSACTIONS_FOR_PREDICTION:
            return sum(t.amount for t in transactions if t.type == 'expense')
        
        df = pd.DataFrame([{
            'amount': t.amount,
            'date': t.date,
            'type': t.type
        } for t in transactions])
        
        expenses = df[df['type'] == 'expense'].copy()
        expenses['day_num'] = (expenses['date'] - expenses['date'].min()).dt.days
        
        if len(expenses) < 2:
            return expenses['amount'].sum()
        
        X = expenses['day_num'].values.reshape(-1, 1)
        y = expenses['amount'].values
        
        self.model.fit(X, y)
        future_day = expenses['day_num'].max() + Config.PREDICTION_DAYS_AHEAD
        predicted = self.model.predict([[future_day]])[0]
        
        return max(0, predicted * 30)
    
    def calculate_stability_score(self, user, transactions):
        if not transactions:
            return 100.0
        
        total_expenses = sum(t.amount for t in transactions if t.type == 'expense')
        total_income = user.monthly_income
        
        if total_income == 0:
            return 0.0
        
        expense_ratio = total_expenses / total_income
        
        if expense_ratio < 0.5:
            score = 100
        elif expense_ratio < 0.7:
            score = 80
        elif expense_ratio < 0.9:
            score = 60
        else:
            score = max(0, 40 - (expense_ratio - 0.9) * 100)
        
        return round(score, 2)
    
    def detect_overspending(self, user, transactions):
        category_spending = self.analyze_spending_pattern(transactions)
        alerts = []
        
        total_expenses = sum(category_spending.values())
        if total_expenses == 0:
            return alerts
        
        for category, amount in category_spending.items():
            percentage = (amount / user.monthly_income) * 100
            
            if category == 'food' and percentage > (Config.FOOD_EXPENSE_THRESHOLD * 100):
                savings = amount - (user.monthly_income * 0.25)
                alerts.append({
                    'category': category,
                    'message': f"Food expenses exceed {Config.FOOD_EXPENSE_THRESHOLD*100:.0f}% of income. Reduce spending to save ₹{savings:.0f}/month",
                    'severity': 'high'
                })
            elif percentage > (Config.CATEGORY_EXPENSE_THRESHOLD * 100):
                alerts.append({
                    'category': category,
                    'message': f"{category.title()} expenses are {percentage:.0f}% of income. Consider reducing.",
                    'severity': 'critical'
                })
        
        return alerts
