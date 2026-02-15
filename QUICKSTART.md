# Quick Start Guide - ML Assignment 2

## 🚀 Get Started in 5 Minutes

### Step 1: Clone/Download (1 min)
```bash
# If using git
git clone <your-repo-url>
cd ml-assignment-2

# Or download ZIP and extract
```

### Step 2: Install Dependencies (2 min)
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Train Models (1 min)
```bash
python train_models.py
```

Expected output:
```
Loading Breast Cancer Wisconsin dataset...
Dataset shape: (569, 30)
Training set: (455, 30)
Test set: (114, 30)

Logistic Regression: Accuracy: 0.9825
Decision Tree: Accuracy: 0.9298
...
All models saved to 'models/' directory
```

### Step 4: Run Streamlit App (1 min)
```bash
streamlit run app.py
```

Browser will open automatically at http://localhost:8501

### Step 5: Test the App
1. Select a model from dropdown
2. Upload `sample_test_data.csv`
3. View predictions and metrics!

---

## 📝 What's in the Project?

```
ml-assignment-2/
│
├── app.py                    # 👈 Main Streamlit app - START HERE
├── train_models.py           # Model training script
├── requirements.txt          # Dependencies
├── README.md                 # Full documentation
│
├── models/                   # Trained models (created after Step 3)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
└── sample_test_data.csv      # Sample data for testing
```

---

## 🎯 Common Tasks

### Run App Locally
```bash
streamlit run app.py
```

### Re-train Models
```bash
python train_models.py
```

### Check Dependencies
```bash
pip list
```

### Freeze New Dependencies
```bash
pip freeze > requirements.txt
```

---

## 🐛 Troubleshooting

### ImportError: No module named 'streamlit'
```bash
pip install -r requirements.txt
```

### Models not found
```bash
# Train models first
python train_models.py
```

### Port already in use
```bash
# Kill existing streamlit process or use different port
streamlit run app.py --server.port 8502
```

### ModuleNotFoundError: No module named 'xgboost'
```bash
pip install xgboost
```

---

## 📦 Deployment to Streamlit Cloud

### Quick Deploy
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repo
4. Deploy!

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

---

## 📊 Dataset Info

**Name**: Breast Cancer Wisconsin (Diagnostic)

**Size**: 
- 569 samples
- 30 features
- 2 classes (Malignant/Benign)

**Source**: Included in scikit-learn

---

## ✅ Assignment Checklist

Before submission:

- [ ] Code runs on BITS Virtual Lab
- [ ] Screenshot taken
- [ ] All 6 models implemented
- [ ] Metrics calculated correctly
- [ ] Streamlit app working locally
- [ ] GitHub repo created
- [ ] Code pushed to GitHub
- [ ] App deployed on Streamlit Cloud
- [ ] README.md complete
- [ ] Submission PDF prepared

---

## 🆘 Need Help?

1. **Check README.md** - Full documentation
2. **Check DEPLOYMENT_GUIDE.md** - Deployment help
3. **Check code comments** - Inline documentation
4. **Test locally first** - Before deploying

---

## 📝 Notes

- **Dataset**: Uses Breast Cancer Wisconsin dataset from sklearn
- **Models**: 6 classification models implemented
- **Deployment**: Free on Streamlit Community Cloud
- **Time**: ~30 minutes total (setup + deploy)

---

**Good luck with your assignment! 🎓**
