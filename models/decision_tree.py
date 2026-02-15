"""
Decision Tree Classifier - ML Assignment 2
BITS Pilani M.Tech (AIML/DSE)
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle


class DecisionTreeModel:
    """
    Decision Tree Model Wrapper
    """
    
    def __init__(self):
        """Initialize Decision Tree with optimal parameters"""
        self.model = DecisionTreeClassifier(
            ccp_alpha=0.0,                      # Complexity parameter
            class_weight=None,                  # No class weights
            criterion='gini',                   # Gini impurity
            max_depth=None,                     # No maximum depth limit
            max_features=None,                  # Consider all features
            max_leaf_nodes=None,                # No limit on leaf nodes
            min_impurity_decrease=0.0,          # Min impurity decrease
            min_samples_leaf=1,                 # Min samples per leaf
            min_samples_split=2,                # Min samples to split
            min_weight_fraction_leaf=0.0,       # Min weight fraction
            random_state=42,                    # Random seed
            splitter='best'                     # Best split strategy
        )
        
    def fit(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        
        print("✓ Decision Tree trained successfully")
        print(f"  Max depth: {self.model.get_depth()}")
        print(f"  Number of leaves: {self.model.get_n_leaves()}")
        
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Predict class probabilities"""
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self):
        """Get feature importances"""
        return self.model.feature_importances_
    
    def save(self, filepath='models/decision_tree.pkl'):
        """Save model to file"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='models/decision_tree.pkl'):
        """Load model from file"""
        instance = cls()
        
        with open(filepath, 'rb') as f:
            instance.model = pickle.load(f)
            
        print(f"✓ Model loaded from {filepath}")
        return instance


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print("="*60)
    print("DECISION TREE - EXAMPLE")
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
    dt_model = DecisionTreeModel()
    dt_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt_model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    dt_model.save()
