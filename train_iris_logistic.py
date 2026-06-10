import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("iris.csv")

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[features]
y = df["Species"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr = LogisticRegression(random_state=42, max_iter=200)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nCoefficients (per feature per class):")
coef_df = pd.DataFrame(lr.coef_, columns=features, index=lr.classes_)
print(coef_df)
