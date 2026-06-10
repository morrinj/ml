import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

k_values = list(range(1, 31))
accs, prec0, prec1, rec0, rec1 = [], [], [], [], []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    accs.append(accuracy_score(y_val, y_pred))
    prec0.append(precision_score(y_val, y_pred, pos_label=0))
    prec1.append(precision_score(y_val, y_pred, pos_label=1))
    rec0.append(recall_score(y_val, y_pred, pos_label=0))
    rec1.append(recall_score(y_val, y_pred, pos_label=1))

results = pd.DataFrame({
    "k": k_values,
    "Accuracy": accs,
    "Precision_Died": prec0,
    "Precision_Survived": prec1,
    "Recall_Died": rec0,
    "Recall_Survived": rec1,
})

print("Metrics for each k (1-30):")
print(results.to_string(index=False))

best_idx = np.argmax(accs)
print(f"\nBest k = {k_values[best_idx]} with Accuracy = {accs[best_idx]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_values, accs, marker="o", label="Accuracy")
plt.plot(k_values, prec1, marker="s", label="Precision (Survived)")
plt.plot(k_values, rec1, marker="^", label="Recall (Survived)")
plt.axvline(x=k_values[best_idx], color="gray", linestyle="--", alpha=0.5)
plt.xticks(range(1, 31))
plt.xlabel("k (number of neighbors)")
plt.ylabel("Score")
plt.title("Effect of k on KNN Performance (Titanic)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("knn_k_analysis.png", dpi=150)
print("\nPlot saved to knn_k_analysis.png")
