"""
Hybrid CNN Model for Financial Prediction
Combines CNN with Dense layers for pattern recognition
Trained on 250,000+ samples
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Install with: pip install tensorflow")


class HybridCNNModel:
    """Hybrid CNN model for financial prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.history = None
        
    def generate_synthetic_dataset(self, n_samples=250000):
        """Generate synthetic financial dataset"""
        np.random.seed(42)
        
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
        
        # Calculate target
        df['total_expense'] = (
            df['food_expense'] + df['transport_expense'] + 
            df['entertainment_expense'] + df['utilities_expense'] + 
            df['healthcare_expense'] + df['education_expense'] +
            df['shopping_expense'] + df['other_expense']
        )
        
        # Add temporal patterns
        df['expense_ratio'] = df['total_expense'] / df['monthly_income']
        df['savings_rate'] = 1 - df['expense_ratio']
        
        return df
    
    def build_hybrid_cnn_model(self, input_shape):
        """Build Hybrid CNN architecture"""
        model = models.Sequential([
            # Reshape for CNN
            layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
            
            # CNN layers for pattern extraction
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            # Dense layers for regression
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)  # Output layer
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
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
    
    def train(self, n_samples=250000, epochs=50, batch_size=256, csv_path=None):
        """Train Hybrid CNN model"""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Cannot train CNN model.")
            return None
        
        print(f"\n{'='*60}")
        print(f"Training Hybrid CNN Model on {n_samples:,} samples")
        print(f"{'='*60}\n")
        
        # Load or generate dataset
        if csv_path:
            print(f"Loading dataset from {csv_path}...")
            df = pd.read_csv(csv_path)
            if n_samples < len(df):
                df = df.sample(n=n_samples, random_state=42)
            print(f"✓ Loaded {len(df):,} samples\n")
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
        
        # Build model
        print("Building Hybrid CNN architecture...")
        self.model = self.build_hybrid_cnn_model(X_train_scaled.shape[1])
        print(f"✓ Model built with {self.model.count_params():,} parameters\n")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            'models/hybrid_cnn_best.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train model
        print("Training Hybrid CNN model...")
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        # Evaluate
        print(f"\n{'='*60}")
        print("Model Evaluation Results")
        print(f"{'='*60}")
        
        y_pred = self.model.predict(X_test_scaled, verbose=0)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nHybrid CNN Performance:")
        print(f"  MAE:  ₹{mae:.2f}")
        print(f"  MSE:  {mse:.2f}")
        print(f"  RMSE: ₹{rmse:.2f}")
        print(f"  R²:   {r2:.4f}")
        
        self.is_trained = True
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'history': self.history.history
        }
    
    def predict(self, features):
        """Predict using trained model"""
        if not self.is_trained or self.model is None:
            return 0
        
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled, verbose=0)
        return prediction[0][0]
    
    def save_model(self, path='models/'):
        """Save trained model"""
        os.makedirs(path, exist_ok=True)
        
        if self.model:
            self.model.save(f'{path}hybrid_cnn_model.h5')
        
        with open(f'{path}cnn_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✓ Hybrid CNN model saved to {path}")
    
    def load_model(self, path='models/'):
        """Load trained model"""
        if not TF_AVAILABLE:
            return False
        
        try:
            self.model = keras.models.load_model(f'{path}hybrid_cnn_model.h5')
            
            with open(f'{path}cnn_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            print(f"✓ Hybrid CNN model loaded from {path}")
            return True
        except:
            return False


# Training script
if __name__ == "__main__":
    if TF_AVAILABLE:
        model = HybridCNNModel()
        
        # Train on 250,000+ samples
        results = model.train(n_samples=250000, epochs=50, batch_size=256)
        
        # Save model
        model.save_model()
        
        # Test prediction
        print("\nTesting Prediction:")
        test_features = [50000, 8000, 5000, 3000, 4000, 2000, 3000, 6000, 2000, 30, 50, 1500]
        prediction = model.predict(test_features)
        print(f"Predicted Total Expense: ₹{prediction:.2f}")
    else:
        print("Install TensorFlow to train CNN model: pip install tensorflow")
