import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features].copy()
y = df["Survived"]

X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})
X = X.fillna(X.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nCoefficients:")
for name, coef in zip(features, lr.coef_[0]):
    print(f"  {name}: {coef:.4f}")

test = pd.read_csv("test.csv")
X_test = test[features].copy()
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})
X_test["Embarked"] = X_test["Embarked"].map({"S": 0, "C": 1, "Q": 2})
X_test = X_test.fillna(X.median(numeric_only=True))
X_test_scaled = scaler.transform(X_test)

test_pred = lr.predict(X_test_scaled)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
submission.to_csv("submission_lr.csv", index=False)
print("\nPredictions saved to submission_lr.csv")
