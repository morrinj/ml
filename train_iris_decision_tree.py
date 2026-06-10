import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[features]
y = df["Species"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nFeature Importances:")
for name, imp in zip(features, clf.feature_importances_):
    print(f"  {name}: {imp:.4f}")

plt.figure(figsize=(16, 8))
plot_tree(clf, filled=True, feature_names=features, class_names=clf.classes_, rounded=True)
plt.savefig("iris_decision_tree.png", dpi=150)
print("\nTree saved to iris_decision_tree.png")
