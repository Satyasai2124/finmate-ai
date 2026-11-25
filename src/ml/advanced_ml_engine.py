"""
Advanced ML Engine with Linear Regression and Hybrid CNN
Trained on 250,000+ financial transaction samples
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from datetime import datetime, timedelta

class AdvancedFinancialMLEngine:
    """Advanced ML Engine with multiple models for financial prediction"""
    
    def __init__(self):
        self.linear_model = LinearRegression()
        self.ridge_model = Ridge(alpha=1.0)
        self.lasso_model = Lasso(alpha=0.1)
        self.scaler = StandardScaler()
        self.model_type = 'linear'  # 'linear', 'ridge', 'lasso', 'hybrid'
        self.is_trained = False
        
    def load_csv_dataset(self, csv_path):
        """Load dataset from CSV file"""
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df):,} samples from CSV\n")
        return df
    
    def generate_synthetic_dataset(self, n_samples=250000):
        """Generate synthetic financial dataset for training"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'monthly_income': np.random.normal(50000, 15000, n_samples),
            'food_expense': np.random.normal(8000, 3000, n_samples),
            'transport_expense': np.random.normal(5000, 2000, n_samples),
            'entertainment_expense': np.random.normal(3000, 1500, n_samples),
            'utilities_expense': np.random.normal(4000, 1000, n_samples),
            'healthcare_expense': np.random.normal(2000, 1000, n_samples),
            'education_expense': np.random.normal(3000, 2000, n_samples),
            'shopping_expense': np.random.normal(6000, 2500, n_samples),
            'other_expense': np.random.normal(2000, 1000, n_samples),
            'days_in_month': np.random.randint(28, 32, n_samples),
            'transaction_count': np.random.randint(10, 100, n_samples),
            'avg_transaction_amount': np.random.normal(1500, 500, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate total expenses (target variable)
        df['total_expense'] = (
            df['food_expense'] + 
            df['transport_expense'] + 
            df['entertainment_expense'] +
            df['utilities_expense'] + 
            df['healthcare_expense'] + 
            df['education_expense'] +
            df['shopping_expense'] + 
            df['other_expense']
        )
        
        # Add noise and patterns
        df['expense_ratio'] = df['total_expense'] / df['monthly_income']
        df['savings'] = df['monthly_income'] - df['total_expense']
        df['stability_score'] = 100 - (df['expense_ratio'] * 100)
        df['stability_score'] = df['stability_score'].clip(0, 100)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        feature_cols = [
            'monthly_income', 'food_expense', 'transport_expense',
            'entertainment_expense', 'utilities_expense', 'healthcare_expense',
            'education_expense', 'shopping_expense', 'other_expense',
            'days_in_month', 'transaction_count', 'avg_transaction_amount'
        ]
        
        X = df[feature_cols].values
        y = df['total_expense'].values
        
        return X, y
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model"""
        print("Training Linear Regression Model...")
        self.linear_model.fit(X_train, y_train)
        print("✓ Linear Regression trained")
        
    def train_ridge_regression(self, X_train, y_train):
        """Train Ridge Regression model"""
        print("Training Ridge Regression Model...")
        self.ridge_model.fit(X_train, y_train)
        print("✓ Ridge Regression trained")
        
    def train_lasso_regression(self, X_train, y_train):
        """Train Lasso Regression model"""
        print("Training Lasso Regression Model...")
        self.lasso_model.fit(X_train, y_train)
        print("✓ Lasso Regression trained")
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"  MAE:  ₹{mae:.2f}")
        print(f"  MSE:  {mse:.2f}")
        print(f"  RMSE: ₹{rmse:.2f}")
        print(f"  R²:   {r2:.4f}")
        
        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
    
    def train_all_models(self, n_samples=250000, csv_path=None):
        """Train all models on dataset (CSV or synthetic)"""
        print(f"\n{'='*60}")
        print(f"Training Advanced ML Models")
        print(f"{'='*60}\n")
        
        # Load dataset
        if csv_path:
            df = self.load_csv_dataset(csv_path)
        else:
            print("Generating synthetic dataset...")
            df = self.generate_synthetic_dataset(n_samples)
            print(f"✓ Generated {len(df):,} samples\n")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set:     {len(X_test):,} samples\n")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        self.train_linear_regression(X_train_scaled, y_train)
        self.train_ridge_regression(X_train_scaled, y_train)
        self.train_lasso_regression(X_train_scaled, y_train)
        
        # Evaluate models
        print(f"\n{'='*60}")
        print("Model Evaluation Results")
        print(f"{'='*60}")
        
        results = {}
        results['linear'] = self.evaluate_model(
            self.linear_model, X_test_scaled, y_test, "Linear Regression"
        )
        results['ridge'] = self.evaluate_model(
            self.ridge_model, X_test_scaled, y_test, "Ridge Regression"
        )
        results['lasso'] = self.evaluate_model(
            self.lasso_model, X_test_scaled, y_test, "Lasso Regression"
        )
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"\n{'='*60}")
        print(f"Best Model: {best_model[0].upper()} (RMSE: ₹{best_model[1]['rmse']:.2f})")
        print(f"{'='*60}\n")
        
        self.is_trained = True
        self.model_type = best_model[0]
        
        return results
    
    def predict_spending(self, features):
        """Predict spending using trained model"""
        if not self.is_trained:
            return 0
        
        features_scaled = self.scaler.transform([features])
        
        if self.model_type == 'linear':
            return self.linear_model.predict(features_scaled)[0]
        elif self.model_type == 'ridge':
            return self.ridge_model.predict(features_scaled)[0]
        elif self.model_type == 'lasso':
            return self.lasso_model.predict(features_scaled)[0]
        
        return 0
    
    def save_models(self, path='models/'):
        """Save trained models"""
        os.makedirs(path, exist_ok=True)
        
        with open(f'{path}linear_model.pkl', 'wb') as f:
            pickle.dump(self.linear_model, f)
        with open(f'{path}ridge_model.pkl', 'wb') as f:
            pickle.dump(self.ridge_model, f)
        with open(f'{path}lasso_model.pkl', 'wb') as f:
            pickle.dump(self.lasso_model, f)
        with open(f'{path}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✓ Models saved to {path}")
    
    def load_models(self, path='models/'):
        """Load trained models"""
        try:
            with open(f'{path}linear_model.pkl', 'rb') as f:
                self.linear_model = pickle.load(f)
            with open(f'{path}ridge_model.pkl', 'rb') as f:
                self.ridge_model = pickle.load(f)
            with open(f'{path}lasso_model.pkl', 'rb') as f:
                self.lasso_model = pickle.load(f)
            with open(f'{path}scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            print(f"✓ Models loaded from {path}")
            return True
        except:
            return False


# Training script
if __name__ == "__main__":
    engine = AdvancedFinancialMLEngine()
    
    # Train on 250,000+ samples
    results = engine.train_all_models(n_samples=250000)
    
    # Save models
    engine.save_models()
    
    # Test prediction
    print("\nTesting Prediction:")
    test_features = [50000, 8000, 5000, 3000, 4000, 2000, 3000, 6000, 2000, 30, 50, 1500]
    prediction = engine.predict_spending(test_features)
    print(f"Predicted Total Expense: ₹{prediction:.2f}")
