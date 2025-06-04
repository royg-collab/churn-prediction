import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Customer Churn Prediction with Data Visualization")

uploaded_file = st.file_uploader("WA_Fn-UseC_-Telco-Customer-Churn.csv", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(data.head())

    # Show basic stats
    st.write("### Data Summary")
    st.write(data.describe())

    # Show churn distribution
    st.write("### Churn Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Churn', data=data, ax=ax1)
    st.pyplot(fig1)

    # Correlation heatmap (only numeric columns)
    st.write("### Correlation Heatmap")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # Prepare data for modeling
    X = data.drop('Churn', axis=1)
    y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Handle categorical variables (simple example: get_dummies)
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy:.2f}")

    # Confusion matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    st.pyplot(fig3)

    # Classification report
    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # User input for new prediction
    st.sidebar.header("Predict churn for a new customer")

    user_input = {}
    for col in X.columns:
        val = st.sidebar.number_input(f"Input value for {col}", value=0)
        user_input[col] = val
    input_df = pd.DataFrame([user_input])

    if st.sidebar.button("Predict"):
        pred = model.predict(input_df)[0]
        result = "Churn" if pred == 1 else "Not Churn"
        st.sidebar.write(f"Prediction result: **{result}**")
