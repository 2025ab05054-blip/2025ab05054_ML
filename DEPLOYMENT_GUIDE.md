# Deployment Guide for Streamlit Community Cloud

## Pre-Deployment Checklist

✅ Complete this checklist before deploying:

1. **Code is working locally**
   ```bash
   streamlit run app.py
   ```

2. **All dependencies in requirements.txt**
   - Verify all imports are listed
   - Check version compatibility

3. **Models are trained and saved**
   ```bash
   python train_models.py
   ```

4. **GitHub repository is ready**
   - All files committed
   - Repository is public
   - .gitignore configured

## Step-by-Step Deployment

### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: ML Classification App"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/ml-assignment-2.git

# Push to GitHub
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Click "Sign in with GitHub"

2. **Create New App**
   - Click "New app" button
   - Select your GitHub repository
   - Branch: `main`
   - Main file path: `app.py`
   - (Optional) Give your app a custom URL

3. **Advanced Settings** (Optional)
   - Python version: 3.8 or higher
   - You can add secrets if needed

4. **Deploy**
   - Click "Deploy!"
   - Wait 2-5 minutes for deployment

### Step 3: Verify Deployment

Your app will be live at:
```
https://[your-app-name].streamlit.app
```

Test the following:
- ✅ App loads without errors
- ✅ Model selection works
- ✅ File upload functions
- ✅ Predictions display correctly
- ✅ Metrics show properly
- ✅ Confusion matrix renders

## Troubleshooting Common Issues

### Issue: ModuleNotFoundError

**Solution**: Add missing package to requirements.txt
```bash
# Locally test:
pip install -r requirements.txt
```

### Issue: Model files not found

**Solution**: Ensure models/ directory and all .pkl files are committed to GitHub
```bash
git add models/*.pkl
git commit -m "Add model files"
git push
```

### Issue: App crashes on startup

**Solution**: Check Streamlit Cloud logs
1. Go to your app dashboard
2. Click on your app
3. Click "Manage app"
4. Check logs for errors

### Issue: Memory limit exceeded

**Solution**: 
- Use smaller model files
- Don't load large datasets in the app
- Use @st.cache_resource decorator

## Updating Your Deployed App

Streamlit Cloud auto-deploys on push to main branch:

```bash
# Make changes locally
# Test locally
streamlit run app.py

# Commit and push
git add .
git commit -m "Update: description of changes"
git push

# Wait 1-2 minutes for auto-deployment
```

## Important Notes

1. **Free Tier Limits**:
   - 1 GB RAM
   - 1 CPU core
   - Public apps only
   - Apps sleep after inactivity

2. **File Size Limits**:
   - Individual file: 100 MB
   - Total repo: 1 GB
   - Consider using Git LFS for large models

3. **Secrets Management**:
   - Don't commit API keys
   - Use Streamlit secrets for sensitive data

## Sample Test Data

Include a small sample_test_data.csv (< 1MB) in your repo for demo purposes.

## Final Checklist

Before submitting:
- ✅ GitHub repo link works
- ✅ Streamlit app link opens
- ✅ App loads without errors
- ✅ All features work
- ✅ README.md is complete
- ✅ Screenshot taken (if required)

## Support

If deployment fails:
- Check Streamlit Community Forum
- Review deployment logs
- Verify requirements.txt
- Test locally first

Good luck! 🚀
