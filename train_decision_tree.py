import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features].copy()
y = df["Survived"]

X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})
X = X.fillna(X.median(numeric_only=True))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nFeature Importances:")
for name, imp in zip(features, clf.feature_importances_):
    print(f"  {name}: {imp:.4f}")

# Optional: Save tree visualization
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=features, class_names=["Died", "Survived"], rounded=True)
plt.savefig("decision_tree.png", dpi=150)
print("\nTree visualization saved to decision_tree.png")

# Predict on test set
test = pd.read_csv("test.csv")
X_test = test[features].copy()
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})
X_test["Embarked"] = X_test["Embarked"].map({"S": 0, "C": 1, "Q": 2})
X_test = X_test.fillna(X.median(numeric_only=True))

test_pred = clf.predict(X_test)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")
