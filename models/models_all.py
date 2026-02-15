"""
Model Definitions - All 6 Classification Models
BITS Pilani ML Assignment 2

This module contains Python implementations of all 6 models:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest
6. XGBoost (if available)

Each model can be created, trained, and saved using the functions below.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os


# ============================================================
# 1. LOGISTIC REGRESSION
# ============================================================

def create_logistic_regression():
    """
    Create Logistic Regression model with specified parameters
    
    Returns:
        LogisticRegression: Configured model
    """
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        max_iter=10000,
        n_jobs=None,
        random_state=42,
        solver='lbfgs',
        tol=0.0001,
        verbose=0,
        warm_start=False
    )
    return model


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    print("Training Logistic Regression...")
    model = create_logistic_regression()
    model.fit(X_train, y_train)
    print("✓ Logistic Regression trained")
    return model


# ============================================================
# 2. DECISION TREE
# ============================================================

def create_decision_tree():
    """
    Create Decision Tree Classifier with specified parameters
    
    Returns:
        DecisionTreeClassifier: Configured model
    """
    model = DecisionTreeClassifier(
        ccp_alpha=0.0,
        class_weight=None,
        criterion='gini',
        max_depth=None,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        random_state=42,
        splitter='best'
    )
    return model


def train_decision_tree(X_train, y_train):
    """Train Decision Tree model"""
    print("Training Decision Tree...")
    model = create_decision_tree()
    model.fit(X_train, y_train)
    print("✓ Decision Tree trained")
    return model


# ============================================================
# 3. K-NEAREST NEIGHBORS (kNN)
# ============================================================

def create_knn():
    """
    Create K-Nearest Neighbors Classifier with specified parameters
    
    Returns:
        KNeighborsClassifier: Configured model
    """
    model = KNeighborsClassifier(
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        metric_params=None,
        n_jobs=None,
        n_neighbors=5,
        p=2,
        weights='uniform'
    )
    return model


def train_knn(X_train, y_train):
    """Train K-Nearest Neighbors model"""
    print("Training K-Nearest Neighbors...")
    model = create_knn()
    model.fit(X_train, y_train)
    print("✓ kNN trained")
    return model


# ============================================================
# 4. NAIVE BAYES
# ============================================================

def create_naive_bayes():
    """
    Create Gaussian Naive Bayes Classifier with specified parameters
    
    Returns:
        GaussianNB: Configured model
    """
    model = GaussianNB(
        priors=None,
        var_smoothing=1e-09
    )
    return model


def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model"""
    print("Training Naive Bayes...")
    model = create_naive_bayes()
    model.fit(X_train, y_train)
    print("✓ Naive Bayes trained")
    return model


# ============================================================
# 5. RANDOM FOREST
# ============================================================

def create_random_forest():
    """
    Create Random Forest Classifier with specified parameters
    
    Returns:
        RandomForestClassifier: Configured model
    """
    model = RandomForestClassifier(
        bootstrap=True,
        ccp_alpha=0.0,
        class_weight=None,
        criterion='gini',
        max_depth=None,
        max_features='sqrt',
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=100,
        n_jobs=None,
        oob_score=False,
        random_state=42,
        verbose=0,
        warm_start=False
    )
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("Training Random Forest...")
    model = create_random_forest()
    model.fit(X_train, y_train)
    print("✓ Random Forest trained")
    return model


# ============================================================
# 6. XGBOOST
# ============================================================

def create_xgboost():
    """
    Create XGBoost Classifier with specified parameters
    
    Returns:
        XGBClassifier: Configured model
    """
    try:
        from xgboost import XGBClassifier
        
        model = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        return model
    except ImportError:
        print("⚠️ XGBoost not installed. Install with: pip install xgboost")
        return None


def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    print("Training XGBoost...")
    model = create_xgboost()
    if model is not None:
        model.fit(X_train, y_train)
        print("✓ XGBoost trained")
    return model


# ============================================================
# STANDARD SCALER
# ============================================================

def create_scaler():
    """
    Create Standard Scaler for feature normalization
    
    Returns:
        StandardScaler: Configured scaler
    """
    scaler = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True
    )
    return scaler


def fit_scaler(X_train):
    """Fit and return scaler"""
    scaler = create_scaler()
    scaler.fit(X_train)
    return scaler


# ============================================================
# SAVE/LOAD UTILITIES
# ============================================================

def save_model(model, filepath):
    """
    Save model to pickle file
    
    Args:
        model: Trained model
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved to {filepath}")


def load_model(filepath):
    """
    Load model from pickle file
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Loaded from {filepath}")
    return model


# ============================================================
# TRAIN ALL MODELS
# ============================================================

def train_all_models(X_train, y_train, X_train_scaled, save_dir='models'):
    """
    Train all 6 models and save them
    
    Args:
        X_train: Training features (unscaled)
        y_train: Training labels
        X_train_scaled: Training features (scaled)
        save_dir: Directory to save models
        
    Returns:
        dict: Dictionary of trained models
    """
    models = {}
    
    # Models that use scaled data
    models['Logistic Regression'] = train_logistic_regression(X_train_scaled, y_train)
    models['kNN'] = train_knn(X_train_scaled, y_train)
    models['Naive Bayes'] = train_naive_bayes(X_train_scaled, y_train)
    
    # Models that use unscaled data
    models['Decision Tree'] = train_decision_tree(X_train, y_train)
    models['Random Forest'] = train_random_forest(X_train, y_train)
    models['XGBoost'] = train_xgboost(X_train, y_train)
    
    # Save all models
    os.makedirs(save_dir, exist_ok=True)
    
    save_model(models['Logistic Regression'], f'{save_dir}/logistic_regression.pkl')
    save_model(models['Decision Tree'], f'{save_dir}/decision_tree.pkl')
    save_model(models['kNN'], f'{save_dir}/knn.pkl')
    save_model(models['Naive Bayes'], f'{save_dir}/naive_bayes.pkl')
    save_model(models['Random Forest'], f'{save_dir}/random_forest.pkl')
    if models['XGBoost'] is not None:
        save_model(models['XGBoost'], f'{save_dir}/xgboost.pkl')
    
    return models


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    """
    Example usage with sample data
    Replace this with your actual dataset
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("="*60)
    print("ML ASSIGNMENT 2 - MODEL TRAINING")
    print("="*60)
    print()
    
    # Generate sample data (REPLACE WITH YOUR ACTUAL DATA)
    print("Generating sample data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print()
    
    # Create and fit scaler
    print("Fitting scaler...")
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    save_model(scaler, 'models/scaler.pkl')
    print()
    
    # Train all models
    print("="*60)
    print("TRAINING ALL MODELS")
    print("="*60)
    print()
    
    models = train_all_models(X_train, y_train, X_train_scaled)
    
    print()
    print("="*60)
    print("EVALUATING MODELS")
    print("="*60)
    print()
    
    # Evaluate each model
    from sklearn.metrics import accuracy_score, classification_report
    
    for name, model in models.items():
        if model is None:
            continue
            
        # Use scaled data for appropriate models
        if name in ['Logistic Regression', 'kNN', 'Naive Bayes']:
            X_test_use = X_test_scaled
        else:
            X_test_use = X_test
        
        y_pred = model.predict(X_test_use)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print()
    
    print("="*60)
    print("ALL MODELS TRAINED AND SAVED!")
    print("="*60)
    print()
    print("Saved models:")
    print("  - models/logistic_regression.pkl")
    print("  - models/decision_tree.pkl")
    print("  - models/knn.pkl")
    print("  - models/naive_bayes.pkl")
    print("  - models/random_forest.pkl")
    print("  - models/xgboost.pkl")
    print("  - models/scaler.pkl")
