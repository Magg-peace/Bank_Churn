# Streamlit Cloud Deployment Instructions

## 🚀 Deploy to Streamlit Cloud

### Prerequisites
- GitHub repository is already set up and code is pushed ✅
- You have a Streamlit Cloud account (free at https://streamlit.io/cloud)

### Deployment Steps:

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Select your repository: `starsgiri/Meghana_S_BDA`
   - Branch: `main`
   - Main file path: `streamlit_app.py`

3. **Advanced Settings (Important!)**
   - Click "Advanced settings"
   - Set Python version: `3.11` or `3.12`
   - **Note:** Models are not included in the repository (too large)
   - The app will show a message that models need to be trained first

4. **Deploy**
   - Click "Deploy!"
   - Wait 2-5 minutes for deployment

### ⚠️ Important Note About Models

The trained models (`models/` folder) are **NOT** included in the GitHub repository because they are too large (>100MB each). 

**For Streamlit Cloud deployment, you have two options:**

#### Option A: Demo Mode (Quick)
The app will run in "demo mode" showing the UI and model training instructions.

#### Option B: Full Deployment with Models (Recommended for Production)
1. Train models on a machine with sufficient resources
2. Upload models to cloud storage (AWS S3, Google Cloud Storage, etc.)
3. Modify `streamlit_app.py` to download models from cloud storage on startup
4. Use Streamlit Secrets for cloud storage credentials

### Example Cloud Storage Integration

```python
# Add to streamlit_app.py before loading models
import streamlit as st
import boto3  # for AWS S3

@st.cache_resource
def download_models_from_s3():
    s3 = boto3.client('s3',
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
        aws_secret_access_key=st.secrets["AWS_SECRET_KEY"]
    )
    # Download models
    s3.download_file('your-bucket', 'models/best_churn_model_v2', 'models/best_churn_model_v2')
```

### Local Testing
Before deploying, test locally:
```bash
cd /home/giri/Desktop/Meghana_S_BDA
source venv/bin/activate
streamlit run streamlit_app.py
```

### Current Deployment Status

✅ Code pushed to GitHub  
✅ Requirements.txt configured  
✅ Streamlit app ready  
⏳ Models need to be uploaded to cloud storage (optional)

### Accessing Your Deployed App

Once deployed, your app will be available at:
`https://[your-app-name].streamlit.app`

Example: `https://meghana-s-bda-churn-prediction.streamlit.app`

---

## 📊 Project Overview

This project implements a Bank Customer Churn Prediction system using:
- **PySpark 3.5.0** for distributed data processing
- **MLlib** for machine learning (Logistic Regression, Random Forest, GBT)
- **Streamlit** for web application interface
- **Dual-version models:** V1 (10K records) and V2 (500K records)

### Features:
- Single customer prediction
- Batch CSV upload for predictions
- Model insights and performance metrics
- Version comparison (V1 vs V2)
