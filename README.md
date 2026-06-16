# ❤️ Heart Disease Risk Predictor
## Week 5 — ML Deployment Project

### 🚀 Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

3. Open browser: http://localhost:8501

### 📁 Project Structure
```
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── ml_model/
│   ├── heart_disease_model.joblib      # Trained ML model
│   ├── heart_disease_model.pkl         # Pickle backup
│   ├── preprocessor.joblib             # Preprocessing pipeline
│   ├── feature_columns.json            # Feature list
│   └── metadata.json                   # Model metrics & config
```

### 📊 Model Performance
- **Model**: Random Forest
- **Test Accuracy**: 83.6%
- **ROC-AUC**: 0.898
- **F1 Score**: 86.5%

### ⚠️ Disclaimer
For educational purposes only. Not a medical diagnosis tool.
