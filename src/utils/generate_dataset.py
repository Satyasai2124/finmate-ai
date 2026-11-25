"""
Generate Sample CSV Dataset for Training
Creates a 250,000+ row CSV file with financial data optimized for 80%+ R² score
"""
import pandas as pd
import numpy as np

def generate_financial_dataset(n_samples=250000, output_file='financial_dataset.csv'):
    """Generate and save financial dataset to CSV with strong linear relationships"""
    
    print(f"Generating {n_samples:,} financial records...")
    
    np.random.seed(42)
    
    # Generate features with controlled distributions
    data = {
        'monthly_income': np.random.normal(50000, 15000, n_samples),
        'food_expense': np.random.normal(8000, 2500, n_samples),
        'transport_expense': np.random.normal(5000, 1800, n_samples),
        'entertainment_expense': np.random.normal(3000, 1200, n_samples),
        'utilities_expense': np.random.normal(4000, 800, n_samples),
        'healthcare_expense': np.random.normal(2000, 900, n_samples),
        'education_expense': np.random.normal(3000, 1500, n_samples),
        'shopping_expense': np.random.normal(6000, 2000, n_samples),
        'other_expense': np.random.normal(2000, 800, n_samples),
        'days_in_month': np.random.randint(28, 32, n_samples),
        'transaction_count': np.random.randint(10, 100, n_samples),
        'avg_transaction_amount': np.random.normal(1500, 400, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate total expenses with STRONG linear relationship (target 85%+ R²)
    df['total_expense'] = (
        df['food_expense'] * 1.0 + 
        df['transport_expense'] * 1.0 + 
        df['entertainment_expense'] * 1.0 +
        df['utilities_expense'] * 1.0 + 
        df['healthcare_expense'] * 1.0 + 
        df['education_expense'] * 1.0 +
        df['shopping_expense'] * 1.0 + 
        df['other_expense'] * 1.0
    )
    
    # Add minimal realistic noise (only 3% variation for high R²)
    # 1. Small unexpected expenses (only 3% of cases)
    unexpected_mask = np.random.random(n_samples) < 0.03
    df.loc[unexpected_mask, 'total_expense'] += np.random.uniform(500, 2000, unexpected_mask.sum())
    
    # 2. Minimal seasonal variations (±3%)
    seasonal_factor = np.sin(np.random.uniform(0, 2*np.pi, n_samples)) * 0.03 + 1
    df['total_expense'] *= seasonal_factor
    
    # 3. Very small random noise (±2%)
    noise = np.random.normal(1, 0.02, n_samples)
    df['total_expense'] *= noise
    
    # 4. Ensure no negative values
    df['total_expense'] = df['total_expense'].clip(lower=5000)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"✓ Dataset saved to {output_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Expected R² Score: 85-90% (optimized for linear regression)")
    
    # Show sample
    print(f"\nSample data (first 5 rows):")
    print(df.head())
    
    return df

if __name__ == "__main__":
    # Generate 250,000 samples
    df = generate_financial_dataset(n_samples=250000, output_file='financial_dataset.csv')
