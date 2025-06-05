# churn-prediction
End-to-end machine learning project to predict customer churn using Random Forest. Includes EDA, model building, insights, Tableau dashboard, and business strategies to reduce churn.

🧠 Customer Churn Prediction with Machine Learning
A complete machine learning pipeline to predict customer churn, extract business insights, and save outputs like confusion matrix, EDA charts, and trained model artifacts for deployment or integration.

🔍 Problem Statement
Customer churn is a key metric for business sustainability. This project predicts whether a customer is likely to churn using historical data from a telecom company. Early detection allows businesses to take corrective action and reduce churn rates.

🛠️ Tools & Technologies
Python

- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn (Random Forest Classifier)
- Joblib (Model Persistence)
- Google Collab / Python Script
- GitHub

📊 Exploratory Data Analysis (EDA)
- Overall churn rate: ~26.5%
- Customers with month-to-month contracts churn more
- Tenure, Internet service type, and contract type show strong correlation with churn

📈 Visualizations saved:
- Confusion matrix.png
- Correlation heatmap.png

🔧 Feature Engineering
- Dropped customerID
- Converted categorical features using get_dummies
- Converted Churn column to binary (Yes = 1, No = 0)
- Ensured consistent feature columns saved (model_columns.joblib)
- Split into train-test (80-20)

🤖 Model Building & Evaluation
Model	Accuracy
- Random Forest- 80%

✅ Final model: Random Forest
✅ Class imbalance handled using class_weight='balanced'
✅ Trained model saved as churn_model.joblib

📌 Confusion Matrix

Also saved in repo as Confusion matrix.png.

📁 Output Artifacts
- churn_model.joblib → Trained ML model
- model_columns.joblib → Feature list for inference
- X_test.csv, y_test.csv → Testing data for validation
= .png → Visualizations for EDA and evaluatio

📈 Business Insights
- Customers with short tenure are much more likely to churn.
- Contract Type, Tech Support, and Monthly Charges are major indicators.
- Targeted offers or improved service in those areas could improve retention.

🚀 Next Steps
Add Streamlit dashboard for prediction
Use SHAP or LIME for interpretability
Schedule retraining with latest data
Deploy model as an API or web app



# 📊 Customer Churn Prediction Dashboard

This project uses a machine learning model to predict telecom customer churn and visualize the insights through a Tableau dashboard.

## 🔧 Tools & Technologies
- Python (pandas, scikit-learn, seaborn, matplotlib)
- Tableau Public
- Joblib (for model saving)

## 🧠 Model
- **Random Forest Classifier** with class_weight='balanced'
- Accuracy: **80%**
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
1. Run `churn2.py` to train and export the model
2. Use Tableau to load `churn_test_result.csv` for visualization

## 📍 Live Demo
- [View on Tableau Public](https://public.tableau.com/views/CustomerChurnDashboard_17490659112280/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

