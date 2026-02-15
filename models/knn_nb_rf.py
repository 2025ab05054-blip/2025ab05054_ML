"""
Remaining Classification Models (kNN, Naive Bayes, Random Forest)
ML Assignment 2 - BITS Pilani M.Tech (AIML/DSE)
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


# ============================================================
# K-NEAREST NEIGHBORS (kNN)
# ============================================================

class KNNModel:
    """K-Nearest Neighbors Model Wrapper"""
    
    def __init__(self):
        """Initialize kNN with optimal parameters"""
        self.model = KNeighborsClassifier(
            algorithm='auto',           # Algorithm to compute nearest neighbors
            leaf_size=30,               # Leaf size for tree algorithms
            metric='minkowski',         # Distance metric
            metric_params=None,         # Additional metric parameters
            n_jobs=None,                # Number of parallel jobs
            n_neighbors=5,              # Number of neighbors
            p=2,                        # Power parameter for Minkowski metric
            weights='uniform'           # Weight function
        )
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """Train the model"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("✓ kNN trained successfully")
        
    def predict(self, X_test):
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def predict_proba(self, X_test):
        """Predict class probabilities"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)
    
    def save(self, filepath='models/knn.pkl'):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        with open(filepath.replace('.pkl', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='models/knn.pkl'):
        """Load model from file"""
        instance = cls()
        with open(filepath, 'rb') as f:
            instance.model = pickle.load(f)
        try:
            with open(filepath.replace('.pkl', '_scaler.pkl'), 'rb') as f:
                instance.scaler = pickle.load(f)
        except:
            pass
        print(f"✓ Model loaded from {filepath}")
        return instance


# ============================================================
# NAIVE BAYES
# ============================================================

class NaiveBayesModel:
    """Gaussian Naive Bayes Model Wrapper"""
    
    def __init__(self):
        """Initialize Naive Bayes with optimal parameters"""
        self.model = GaussianNB(
            priors=None,                # No prior probabilities
            var_smoothing=1e-09         # Variance smoothing
        )
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """Train the model"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("✓ Naive Bayes trained successfully")
        print(f"  Classes: {self.model.classes_}")
        
    def predict(self, X_test):
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def predict_proba(self, X_test):
        """Predict class probabilities"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)
    
    def save(self, filepath='models/naive_bayes.pkl'):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        with open(filepath.replace('.pkl', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='models/naive_bayes.pkl'):
        """Load model from file"""
        instance = cls()
        with open(filepath, 'rb') as f:
            instance.model = pickle.load(f)
        try:
            with open(filepath.replace('.pkl', '_scaler.pkl'), 'rb') as f:
                instance.scaler = pickle.load(f)
        except:
            pass
        print(f"✓ Model loaded from {filepath}")
        return instance


# ============================================================
# RANDOM FOREST
# ============================================================

class RandomForestModel:
    """Random Forest Model Wrapper"""
    
    def __init__(self):
        """Initialize Random Forest with optimal parameters"""
        self.model = RandomForestClassifier(
            bootstrap=True,                     # Bootstrap samples
            ccp_alpha=0.0,                      # Complexity parameter
            class_weight=None,                  # No class weights
            criterion='gini',                   # Gini impurity
            max_depth=None,                     # No max depth
            max_features='sqrt',                # Square root of features
            max_leaf_nodes=None,                # No limit
            max_samples=None,                   # Use all samples
            min_impurity_decrease=0.0,          # Min impurity decrease
            min_samples_leaf=1,                 # Min samples per leaf
            min_samples_split=2,                # Min samples to split
            min_weight_fraction_leaf=0.0,       # Min weight fraction
            n_estimators=100,                   # Number of trees
            n_jobs=None,                        # Number of jobs
            oob_score=False,                    # No out-of-bag score
            random_state=42,                    # Random seed
            verbose=0,                          # No verbose output
            warm_start=False                    # Don't reuse solution
        )
        
    def fit(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        print("✓ Random Forest trained successfully")
        print(f"  Number of trees: {self.model.n_estimators}")
        print(f"  Number of features: {self.model.n_features_in_}")
        
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Predict class probabilities"""
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self):
        """Get feature importances"""
        return self.model.feature_importances_
    
    def save(self, filepath='models/random_forest.pkl'):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='models/random_forest.pkl'):
        """Load model from file"""
        instance = cls()
        with open(filepath, 'rb') as f:
            instance.model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")
        return instance


# ============================================================
# EXAMPLE USAGE FOR ALL MODELS
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print("="*60)
    print("TESTING ALL MODELS (kNN, Naive Bayes, Random Forest)")
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
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print()
    
    # Test kNN
    print("-"*60)
    print("K-NEAREST NEIGHBORS")
    print("-"*60)
    knn_model = KNNModel()
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
    knn_model.save()
    print()
    
    # Test Naive Bayes
    print("-"*60)
    print("NAIVE BAYES")
    print("-"*60)
    nb_model = NaiveBayesModel()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    nb_model.save()
    print()
    
    # Test Random Forest
    print("-"*60)
    print("RANDOM FOREST")
    print("-"*60)
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    rf_model.save()
    print()
    
    print("="*60)
    print("ALL MODELS TESTED AND SAVED!")
    print("="*60)
