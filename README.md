# churn-prediction
End-to-end machine learning project to predict customer churn using Logistic Regression and Random Forest. Includes EDA, model building, insights, Tableau dashboard, and business strategies to reduce churn.

# 📊 Customer Churn Prediction Dashboard

This project uses a machine learning model to predict telecom customer churn and visualize the insights through a Tableau dashboard.

## 🔧 Tools & Technologies
- Python (pandas, scikit-learn, seaborn, matplotlib)
- Tableau Public
- Joblib (for model saving)

## 🧠 Model
- **Random Forest Classifier** with class_weight='balanced'
- Accuracy: **XX%**
- Exported `X_test`, `y_test`, and predictions for dashboard use

## 📊 Dashboard Highlights
- Confusion Matrix Heatmap
- KPI Cards (Accuracy)
- Filters: Contract Type, Gender, Internet Service

## 📸 Preview

![Dashboard](images/dashboard_screenshot.png)

## 📁 Dataset
- Source: [IBM Sample Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

## 🚀 How to Run
1. Run `churn_model_training.py` to train and export the model
2. Use Tableau to load `churn_test_result.csv` for visualization

## 📍 Live Demo
- [View on Tableau Public](https://public.tableau.com/views/CustomerChurnDashboard_17490659112280/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

