"""
Complete Training Pipeline
Trains both Linear Regression and Hybrid CNN models
Compares performance on 250,000+ samples
"""
import sys
import os
from advanced_ml_engine import AdvancedFinancialMLEngine
from hybrid_cnn_model import HybridCNNModel, TF_AVAILABLE

def train_all_models(n_samples=250000, csv_path=None):
    """Train and compare all models"""
    print("\n" + "="*70)
    print(" FINANCIAL PREDICTION MODEL TRAINING")
    if csv_path:
        print(f" Dataset: {csv_path}")
    else:
        print(f" Dataset Size: {n_samples:,} samples (synthetic)")
    print("="*70 + "\n")
    
    results = {}
    
    # Train Linear Regression models
    print("\n[1/2] Training Linear Regression Models...")
    print("-" * 70)
    linear_engine = AdvancedFinancialMLEngine()
    linear_results = linear_engine.train_all_models(n_samples=n_samples, csv_path=csv_path)
    linear_engine.save_models()
    results['linear'] = linear_results
    
    # Train Hybrid CNN model
    if TF_AVAILABLE:
        print("\n[2/2] Training Hybrid CNN Model...")
        print("-" * 70)
        cnn_model = HybridCNNModel()
        cnn_results = cnn_model.train(n_samples=n_samples, epochs=50, batch_size=256)
        cnn_model.save_model()
        results['cnn'] = cnn_results
    else:
        print("\n[2/2] Skipping CNN Model (TensorFlow not installed)")
        print("Install with: pip install tensorflow")
    
    # Compare results
    print("\n" + "="*70)
    print(" MODEL COMPARISON SUMMARY")
    print("="*70 + "\n")
    
    print(f"{'Model':<20} {'MAE (₹)':<15} {'RMSE (₹)':<15} {'R²':<10}")
    print("-" * 70)
    
    for model_name, metrics in linear_results.items():
        print(f"{model_name.upper():<20} {metrics['mae']:<15.2f} {metrics['rmse']:<15.2f} {metrics['r2']:<10.4f}")
    
    if TF_AVAILABLE and 'cnn' in results:
        print(f"{'HYBRID CNN':<20} {cnn_results['mae']:<15.2f} {cnn_results['rmse']:<15.2f} {cnn_results['r2']:<10.4f}")
    
    # Find best model
    all_models = list(linear_results.items())
    if TF_AVAILABLE and 'cnn' in results:
        all_models.append(('hybrid_cnn', cnn_results))
    
    best_model = min(all_models, key=lambda x: x[1]['rmse'])
    
    print("\n" + "="*70)
    print(f" BEST MODEL: {best_model[0].upper()}")
    print(f" RMSE: ₹{best_model[1]['rmse']:.2f}")
    print(f" R²: {best_model[1]['r2']:.4f}")
    print("="*70 + "\n")
    
    return results


def test_predictions():
    """Test predictions with sample data"""
    print("\n" + "="*70)
    print(" TESTING PREDICTIONS")
    print("="*70 + "\n")
    
    # Sample features
    test_cases = [
        {
            'name': 'Low Spender',
            'features': [50000, 5000, 3000, 2000, 3000, 1500, 2000, 4000, 1500, 30, 40, 1200]
        },
        {
            'name': 'Medium Spender',
            'features': [50000, 8000, 5000, 3000, 4000, 2000, 3000, 6000, 2000, 30, 50, 1500]
        },
        {
            'name': 'High Spender',
            'features': [50000, 12000, 8000, 5000, 5000, 3000, 4000, 8000, 3000, 30, 70, 2000]
        }
    ]
    
    # Load models
    linear_engine = AdvancedFinancialMLEngine()
    if linear_engine.load_models():
        print("✓ Linear models loaded\n")
        
        for test_case in test_cases:
            prediction = linear_engine.predict_spending(test_case['features'])
            actual = sum(test_case['features'][1:9])  # Sum of expense features
            print(f"{test_case['name']}:")
            print(f"  Predicted: ₹{prediction:.2f}")
            print(f"  Actual:    ₹{actual:.2f}")
            print(f"  Error:     ₹{abs(prediction - actual):.2f}\n")
    
    if TF_AVAILABLE:
        cnn_model = HybridCNNModel()
        if cnn_model.load_model():
            print("\n✓ CNN model loaded\n")
            
            for test_case in test_cases:
                prediction = cnn_model.predict(test_case['features'])
                actual = sum(test_case['features'][1:9])
                print(f"{test_case['name']} (CNN):")
                print(f"  Predicted: ₹{prediction:.2f}")
                print(f"  Actual:    ₹{actual:.2f}")
                print(f"  Error:     ₹{abs(prediction - actual):.2f}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train financial prediction models')
    parser.add_argument('--samples', type=int, default=250000, help='Number of training samples')
    parser.add_argument('--csv', type=str, default=None, help='Path to CSV dataset file')
    parser.add_argument('--test-only', action='store_true', help='Only test existing models')
    
    args = parser.parse_args()
    
    if args.test_only:
        test_predictions()
    else:
        # Train models
        results = train_all_models(n_samples=args.samples, csv_path=args.csv)
        
        # Test predictions
        test_predictions()
        
        print("\n✓ Training complete! Models saved in 'models/' directory")
        print("\nTo test predictions: python train_models.py --test-only")
