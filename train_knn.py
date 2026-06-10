import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

# Find best k
best_k = 5
best_acc = 0
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_val, knn.predict(X_val))
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"Best k: {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)

acc = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred, output_dict=True)
p0 = report["0"]["precision"]
r0 = report["0"]["recall"]
p1 = report["1"]["precision"]
r1 = report["1"]["recall"]

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: Died={p0:.4f}, Survived={p1:.4f}")
print(f"Recall:    Died={r0:.4f}, Survived={r1:.4f}")

test = pd.read_csv("test.csv")
X_test = test[features].copy()
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})
X_test["Embarked"] = X_test["Embarked"].map({"S": 0, "C": 1, "Q": 2})
X_test = X_test.fillna(X.median(numeric_only=True))
X_test_scaled = scaler.transform(X_test)

test_pred = knn.predict(X_test_scaled)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
submission.to_csv("submission_knn.csv", index=False)
print("\nPredictions saved to submission_knn.csv")
