# ==========================================================
# Customer Churn Prediction using Machine Learning
# Tech: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------------------------------------
# STEP 1: Load the data
# ----------------------------------------------------------
df = pd.read_csv("customer_churn.csv")
print("Rows and columns:", df.shape)
print(df.head())

# ----------------------------------------------------------
# STEP 2: Data cleaning
# ----------------------------------------------------------
# customerID is just a unique label, it does not help prediction
df = df.drop("customerID", axis=1)

# TotalCharges can be stored as text in some datasets -> convert to number
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values and fill them with the median
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Convert target column: Yes -> 1 (churned), No -> 0 (stayed)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ----------------------------------------------------------
# STEP 3: Exploratory Data Analysis (EDA) - understand the data
# ----------------------------------------------------------
print("\nChurn rate: {:.1f}%".format(df["Churn"].mean() * 100))

# Chart 1: how many customers churned?
sns.countplot(x="Churn", data=df)
plt.title("Churn Count (0 = Stayed, 1 = Churned)")
#plt.savefig("outputs/1_churn_count.png")
plt.show()
plt.close()

# Chart 2: churn by contract type
df.groupby("Contract")["Churn"].mean().plot(kind="bar", color="tomato")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn rate")
plt.tight_layout()
#plt.savefig("outputs/2_churn_by_contract.png")
plt.show()
plt.close()

# Chart 3: tenure of churned vs stayed customers
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
#plt.savefig("outputs/3_tenure_vs_churn.png")
plt.show()
plt.close()

# ----------------------------------------------------------
# STEP 4: Data preprocessing
# ----------------------------------------------------------
# ML models only understand numbers, so convert text columns
# (Contract, PaymentMethod ...) into 0/1 columns -> One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# Split into input features (X) and target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 80% data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric columns so that big numbers (TotalCharges) don't dominate
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ----------------------------------------------------------
# STEP 5: Train the models
# ----------------------------------------------------------
# Model 1: Logistic Regression (simple, good baseline)
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train, y_train)

# Model 2: Random Forest (many decision trees combined)
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)

# ----------------------------------------------------------
# STEP 6: Evaluate the models
# ----------------------------------------------------------
def evaluate(name, model):
    pred = model.predict(X_test)
    print("\n----- " + name + " -----")
    print("Accuracy :", round(accuracy_score(y_test, pred), 3))
    print("Precision:", round(precision_score(y_test, pred), 3))
    print("Recall   :", round(recall_score(y_test, pred), 3))
    print("F1 Score :", round(f1_score(y_test, pred), 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    return pred

evaluate("Logistic Regression", log_model)
rf_pred = evaluate("Random Forest", rf_model)
print("\nClassification report (Random Forest):\n", classification_report(y_test, rf_pred))

# Confusion matrix chart for Random Forest
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#plt.savefig("outputs/4_confusion_matrix.png")
plt.close()

# ----------------------------------------------------------
# STEP 7: Which factors affect churn the most? (Feature Importance)
# ----------------------------------------------------------
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
top10 = importance.sort_values(ascending=False).head(10)
print("\nTop 10 factors influencing churn:\n", top10.round(3))

top10.sort_values().plot(kind="barh", color="steelblue")
plt.title("Top 10 Factors Influencing Churn")
plt.tight_layout()
#plt.savefig("outputs/5_feature_importance.png")
plt.show()
plt.close()

# ----------------------------------------------------------
# STEP 8: Predict for ONE new customer (demo)
# ----------------------------------------------------------
new_customer = X_test.iloc[[0]]          # take one customer from test data
prob = rf_model.predict_proba(new_customer)[0][1]
print("\nChurn probability of sample customer: {:.1f}%".format(prob * 100))
print("Prediction:", "Will CHURN" if prob > 0.5 else "Will STAY")
