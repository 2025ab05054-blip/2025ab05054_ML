# Machine Learning Classification Model Comparison

## Problem Statement

This project implements and compares six different machine learning classification algorithms on the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to predict whether a tumor is malignant (cancerous) or benign (non-cancerous) based on various features computed from digitized images of fine needle aspirates (FNA) of breast masses.

**Objective**: Build and deploy an interactive web application that allows users to:
- Compare performance of different classification models
- Upload test data and get predictions
- View comprehensive evaluation metrics
- Analyze confusion matrices and classification reports

---

## Dataset Description

**Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset

**Source**: UCI Machine Learning Repository / scikit-learn datasets

**Description**: 
The dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses. These features describe characteristics of cell nuclei present in the images.

**Dataset Statistics**:
- **Number of Instances**: 569
- **Number of Features**: 30
- **Number of Classes**: 2 (Malignant, Benign)
- **Class Distribution**: 
  - Malignant: 212 (37.3%)
  - Benign: 357 (62.7%)

**Features** (30 total):
For each cell nucleus, ten real-valued features are computed:
1. radius (mean of distances from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter² / area - 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension ("coastline approximation" - 1)

For each of these 10 features, three values are computed:
- Mean
- Standard Error
- "Worst" (mean of the three largest values)

This results in 30 features total.

**Target Variable**:
- 0: Malignant (cancer)
- 1: Benign (non-cancer)

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

*Note: Metrics calculated on test set (20% of data, stratified split)*

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

## Key Insights

1. **Best Model**: **Logistic Regression** surprisingly outperformed all ensemble methods, achieving 98.25% accuracy and 99.86% AUC. This demonstrates that simpler models can excel when the problem has a relatively linear decision boundary.

2. **Ensemble Performance**: Both Random Forest and XGBoost performed well but didn't significantly outperform simpler models on this dataset, likely due to:
   - Small dataset size (569 samples)
   - Well-separated classes
   - Relatively simple decision boundary

3. **Linear Separability**: The strong performance of Logistic Regression and high metrics across most models suggest the classes are nearly linearly separable in the 30-dimensional feature space.

4. **AUC Scores**: All models except Decision Tree achieved AUC > 0.99, indicating excellent class discrimination ability across the board.

5. **Production Recommendation**: For deployment, **Logistic Regression** is recommended due to:
   - Best performance metrics
   - Fast inference time
   - Low memory footprint
   - High interpretability
   - Easy to explain to medical professionals

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd ml-assignment-2
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Train models** (optional - pre-trained models included):
```bash
python train_models.py
```

4. **Run Streamlit app locally**:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Usage

### Running the Application

1. **Select a Model**: Use the dropdown in the sidebar to choose one of the 6 classification models

2. **Upload Test Data**: 
   - Click "Browse files" in the sidebar
   - Upload a CSV file with the same 30 features as the training data
   - Optionally include a 'target' column for evaluation

3. **View Results**:
   - **Predictions**: See predicted classes and probabilities
   - **Metrics**: View accuracy, precision, recall, F1, MCC, and AUC scores
   - **Confusion Matrix**: Visualize prediction performance
   - **Classification Report**: Detailed per-class metrics


---

## Project Structure

```
ml-assignment-2/
│
├── app.py                          # Main Streamlit application
├── train_models.py                 # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/                         # Saved model files
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── sample_test_data.csv           # Sample test data for demo
└── model_comparison_results.csv   # Training results
```

---

## Deployment

### Streamlit Community Cloud

This application is deployed on Streamlit Community Cloud:

**Live App**: [Your Deployed App URL]

To deploy your own version:

1. Push code to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository, branch, and `app.py`
6. Click "Deploy"

---

## Technologies Used

- **Python 3.8+**
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning models and metrics
- **XGBoost**: Gradient boosting framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization

---

## Model Details

### 1. Logistic Regression
- Linear model for binary classification
- Uses sigmoid function for probability estimation
- Regularization: Default (L2)
- Solver: lbfgs
- Max iterations: 10000

### 2. Decision Tree Classifier
- Tree-based model with recursive binary splits
- Criterion: Gini impurity
- No depth limit (default)
- Minimum samples per leaf: 1 (default)

### 3. K-Nearest Neighbors
- Instance-based learning algorithm
- Number of neighbors: 5
- Distance metric: Euclidean
- Weights: Uniform

### 4. Naive Bayes (Gaussian)
- Probabilistic classifier
- Assumes Gaussian distribution
- Features assumed independent
- No hyperparameters tuned

### 5. Random Forest
- Ensemble of 100 decision trees
- Bootstrap aggregating (bagging)
- Feature randomness at each split
- Out-of-bag estimation enabled

### 6. XGBoost
- Gradient boosting framework
- Default hyperparameters
- Loss: Logistic
- Early stopping: Disabled

---

## Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient)**: Balanced measure even for imbalanced classes
## Author

Name: Sathish Reddy VasiReddy
ID : 2025ab05054