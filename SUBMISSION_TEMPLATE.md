# BITS Pilani M.Tech (AIML/DSE)
# Machine Learning - Assignment 2
# Submission Document

---

## Student Information

**Name**: [Your Name]  
**ID Number**: [Your ID]  
**Program**: M.Tech (AIML/DSE)  
**Course**: Machine Learning  
**Assignment**: Assignment 2  
**Submission Date**: [Date]

---

## 1. GitHub Repository Link

**Repository URL**: https://github.com/[YOUR_USERNAME]/ml-assignment-2

**Contents Verification**:
- ✅ app.py (Streamlit application)
- ✅ train_models.py (Model training script)
- ✅ requirements.txt
- ✅ README.md (with all required sections)
- ✅ models/ directory (6 trained models)
- ✅ sample_test_data.csv

---

## 2. Live Streamlit App Link

**Deployed App URL**: https://[your-app-name].streamlit.app

**App Features**:
- ✅ Dataset upload option (CSV)
- ✅ Model selection dropdown
- ✅ Display of evaluation metrics
- ✅ Confusion matrix visualization
- ✅ Classification report

---

## 3. BITS Virtual Lab Screenshot

[INSERT SCREENSHOT HERE]

**Screenshot shows**:
- Terminal/Jupyter Notebook with code execution
- BITS Virtual Lab interface visible
- Assignment code running successfully
- Date/timestamp visible

---

## 4. README Content

[COPY THE ENTIRE README.MD CONTENT BELOW]

---

# Machine Learning Classification Model Comparison

## Problem Statement

This project implements and compares six different machine learning classification algorithms on the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to predict whether a tumor is malignant (cancerous) or benign (non-cancerous) based on various features computed from digitized images of fine needle aspirates (FNA) of breast masses.

## Dataset Description

**Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset

**Source**: UCI Machine Learning Repository / scikit-learn datasets

**Dataset Statistics**:
- **Number of Instances**: 569
- **Number of Features**: 30
- **Number of Classes**: 2 (Malignant, Benign)
- **Class Distribution**: 
  - Malignant: 212 (37.3%)
  - Benign: 357 (62.7%)

**Features** (30 total):
For each cell nucleus, ten real-valued features are computed (mean, standard error, and "worst"):
1. radius, 2. texture, 3. perimeter, 4. area, 5. smoothness
6. compactness, 7. concavity, 8. concave points, 9. symmetry, 10. fractal dimension

**Target Variable**: 0 (Malignant), 1 (Benign)

---

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|------|
| Logistic Regression | 0.9825 | 0.9986 | 0.9827 | 0.9825 | 0.9824 | 0.9628 |
| Decision Tree | 0.9298 | 0.9241 | 0.9315 | 0.9298 | 0.9295 | 0.8503 |
| K-Nearest Neighbors | 0.9649 | 0.9945 | 0.9655 | 0.9649 | 0.9647 | 0.9240 |
| Naive Bayes | 0.9474 | 0.9943 | 0.9490 | 0.9474 | 0.9471 | 0.8870 |
| Random Forest | 0.9737 | 0.9979 | 0.9741 | 0.9737 | 0.9736 | 0.9434 |
| XGBoost | 0.9649 | 0.9967 | 0.9657 | 0.9649 | 0.9647 | 0.9238 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieved the **best overall performance** with highest accuracy (98.25%), AUC (99.86%), and MCC (96.28%). This suggests that the decision boundary between malignant and benign tumors is relatively linear in the feature space. The model generalizes very well despite being a simple linear classifier. Excellent choice for this dataset due to high interpretability and outstanding performance. |
| **Decision Tree** | Showed the **lowest performance** among all models with accuracy of 92.98% and MCC of 85.03%. The model likely overfits to training data and doesn't generalize as well. Decision trees are prone to overfitting when not properly pruned. The significantly lower AUC (92.41%) compared to other models indicates poor probability calibration. However, it offers high interpretability through visualization. |
| **K-Nearest Neighbors** | Performed **strongly** with 96.49% accuracy and high AUC (99.45%). The model benefits from the well-separated clusters in the feature space. Performance depends heavily on choosing the right value of k (5 in this case). Computationally expensive for large datasets as it requires distance calculation to all training samples. Good balance between performance and simplicity. |
| **Naive Bayes** | Achieved **good performance** (94.74% accuracy) despite the strong independence assumption. The relatively high AUC (99.43%) indicates good probability estimates. Performance suggests features are somewhat independent, making Naive Bayes a viable option. Very fast training and prediction, suitable for real-time applications. Lower MCC (88.70%) indicates some false positives/negatives. |
| **Random Forest** | Demonstrated **excellent performance** (97.37% accuracy) with very high AUC (99.79%) and strong MCC (94.34%). The ensemble approach effectively reduces overfitting seen in single decision trees. Robust to outliers and handles feature interactions well. Provides feature importance rankings. Second-best overall performer after Logistic Regression. |
| **XGBoost** | Achieved **strong performance** (96.49% accuracy) with high AUC (99.67%). Surprisingly, it performed similarly to KNN and slightly below Random Forest on this dataset. This might be because the dataset is relatively small and simple, where simpler models can match complex gradient boosting. XGBoost typically excels on larger, more complex datasets. Still provides excellent probability estimates and handles imbalanced classes well. |

---

## Declaration

I hereby declare that:
1. This assignment is my original work
2. I have used the BITS Virtual Lab for execution
3. All code has been written/tested by me
4. Proper citations have been provided where external resources were referenced
5. I understand the academic integrity policy

**Signature**: ________________  
**Date**: ________________

---

**END OF SUBMISSION DOCUMENT**
