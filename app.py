import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification Model Comparison",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Machine Learning Classification Model Comparison")
st.markdown("""
This application demonstrates 6 different classification models trained on the Breast Cancer Wisconsin dataset.
Upload your test data to see predictions and evaluation metrics.
""")

# Sidebar for model selection
st.sidebar.header("Configuration")

# Model selection
models_available = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

selected_model = st.sidebar.selectbox(
    "Select a Model",
    list(models_available.keys())
)

# Load model function
@st.cache_resource
def load_model(model_name):
    model_path = os.path.join("models", models_available[model_name])
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None

# File uploader
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=['csv'],
    help="Upload a CSV file with the same features as the training data"
)

# Main content
if uploaded_file is not None:
    # Read data
    try:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head(10))
        
        st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Separate features and target
        if 'target' in df.columns:
            X_test = df.drop('target', axis=1)
            y_test = df['target']
            has_labels = True
        else:
            X_test = df
            y_test = None
            has_labels = False
            st.warning("⚠️ No 'target' column found. Only predictions will be shown.")
        
        # Load selected model
        model = load_model(selected_model)
        
        if model is not None:
            # Make predictions
            st.subheader(f"🎯 Predictions using {selected_model}")
            
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Show predictions
                pred_df = pd.DataFrame({
                    'Sample': range(len(y_pred)),
                    'Predicted Class': y_pred
                })
                if y_pred_proba is not None:
                    pred_df['Prediction Probability'] = y_pred_proba
                
                st.dataframe(pred_df.head(20))
                
                # Evaluation metrics (if labels are available)
                if has_labels:
                    st.subheader("📈 Evaluation Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    mcc = matthews_corrcoef(y_test, y_pred)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        st.metric("Precision", f"{precision:.4f}")
                    
                    with col2:
                        st.metric("Recall", f"{recall:.4f}")
                        st.metric("F1 Score", f"{f1:.4f}")
                    
                    with col3:
                        st.metric("MCC Score", f"{mcc:.4f}")
                        if y_pred_proba is not None:
                            try:
                                auc = roc_auc_score(y_test, y_pred_proba)
                                st.metric("AUC Score", f"{auc:.4f}")
                            except:
                                st.metric("AUC Score", "N/A")
                    
                    # Confusion Matrix
                    st.subheader("🔢 Confusion Matrix")
                    
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted Label')
                    ax.set_ylabel('True Label')
                    ax.set_title(f'Confusion Matrix - {selected_model}')
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader("📋 Classification Report")
                    
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0))
                    
                else:
                    st.info("ℹ️ Upload data with 'target' column to see evaluation metrics")
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                st.exception(e)
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Please upload a CSV file to begin analysis")
    
    # Show sample data format
    st.subheader("📝 Expected Data Format")
    st.markdown("""
    Your CSV file should contain:
    - All feature columns (same as training data)
    - Optional: 'target' column with actual labels for evaluation
    
    **Example:**
    ```
    feature1, feature2, feature3, ..., target
    1.5, 2.3, 4.1, ..., 0
    2.1, 3.4, 5.2, ..., 1
    ```
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**BITS Pilani M.Tech (AIML/DSE)**  
Machine Learning - Assignment 2  
Submission Deadline: 15-Feb-2026
""")
