"""
Logistic Regression Model - ML Assignment 2
BITS Pilani M.Tech (AIML/DSE)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle


class LogisticRegressionModel:
    """
    Logistic Regression Model Wrapper
    """
    
    def __init__(self):
        """Initialize Logistic Regression with optimal parameters"""
        self.model = LogisticRegression(
            C=1.0,                      # Regularization strength
            class_weight=None,          # No class weights
            dual=False,                 # Primal formulation
            fit_intercept=True,         # Include intercept
            intercept_scaling=1,        # Intercept scaling
            max_iter=10000,             # Maximum iterations
            n_jobs=None,                # Number of CPU cores
            random_state=42,            # Random seed for reproducibility
            solver='lbfgs',             # Optimization algorithm
            tol=0.0001,                 # Tolerance for stopping criteria
            verbose=0,                  # No verbose output
            warm_start=False            # Don't reuse previous solution
        )
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        print("✓ Logistic Regression trained successfully")
        print(f"  Coefficients shape: {self.model.coef_.shape}")
        print(f"  Number of iterations: {self.model.n_iter_}")
        
    def predict(self, X_test):
        """
        Make predictions
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities
        
        Args:
            X_test: Test features
            
        Returns:
            Class probabilities
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)
    
    def save(self, filepath='models/logistic_regression.pkl'):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(filepath.replace('.pkl', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='models/logistic_regression.pkl'):
        """
        Load model from file
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model instance
        """
        instance = cls()
        
        with open(filepath, 'rb') as f:
            instance.model = pickle.load(f)
        
        scaler_path = filepath.replace('.pkl', '_scaler.pkl')
        try:
            with open(scaler_path, 'rb') as f:
                instance.scaler = pickle.load(f)
        except:
            pass
            
        print(f"✓ Model loaded from {filepath}")
        return instance


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print("="*60)
    print("LOGISTIC REGRESSION - EXAMPLE")
    print("="*60)
    print()
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    lr_model = LogisticRegressionModel()
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    lr_model.save()
