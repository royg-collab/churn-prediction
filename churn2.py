import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
data = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Show basic info
print("First 5 rows:\n", data.head())
print("\nSummary stats:\n", data.describe())
print("\nChurn counts:\n", data['Churn'].value_counts())

# Plot churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=data)
plt.title("Churn Distribution")
plt.savefig("churn_distribution.png")
plt.close()

# Correlation heatmap
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(6,4))
sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Convert target to binary
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop customerID
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)

# Convert all categorical columns using get_dummies
data = pd.get_dummies(data)

# Features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model with class_weight='balanced'
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
joblib.dump(model, "churn_model.joblib")
print("Model saved as churn_model.joblib")

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("\nConfusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)
