from datetime import datetime, timedelta

class RecommendationEngine:
    def generate_recommendations(self, user, transactions, stability_score):
        recommendations = []
        
        if not transactions:
            return ["Start tracking your expenses to get personalized recommendations"]
        
        recent_transactions = [t for t in transactions if t.date > datetime.utcnow() - timedelta(days=30)]
        total_expenses = sum(t.amount for t in recent_transactions if t.type == 'expense')
        total_income = user.monthly_income
        
        expense_ratio = total_expenses / total_income if total_income > 0 else 0
        
        if expense_ratio > 0.9:
            recommendations.append("⚠️ Critical: You're spending 90%+ of income. Reduce non-essential expenses immediately.")
        elif expense_ratio > 0.7:
            recommendations.append("⚡ Warning: High spending detected. Aim to save at least 20% of income.")
        
        category_spending = {}
        for t in recent_transactions:
            if t.type == 'expense':
                category_spending[t.category] = category_spending.get(t.category, 0) + t.amount
        
        if category_spending:
            top_category = max(category_spending, key=category_spending.get)
            top_amount = category_spending[top_category]
            top_percentage = (top_amount / total_income) * 100
            
            if top_percentage > 40:
                recommendations.append(f"💡 {top_category.title()} is {top_percentage:.0f}% of income. Try reducing by 20% to save ₹{top_amount*0.2:.0f}")
        
        if 'food' in category_spending:
            food_spending = category_spending['food']
            if food_spending > total_income * 0.3 and 'Hyderabad' in user.location:
                recommendations.append("🍽️ Consider meal prep instead of food delivery to save ₹1,800+/month")
        
        savings = total_income - total_expenses
        if savings > 0:
            recommendations.append(f"✅ Good job! You saved ₹{savings:.0f} this month. Try to maintain this.")
        
        if stability_score < 50:
            recommendations.append("📊 Financial stability is low. Focus on essential expenses only.")
        
        return recommendations if recommendations else ["Keep tracking expenses for better insights"]
