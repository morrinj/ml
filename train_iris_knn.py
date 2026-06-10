import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[features]
y = df["Species"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

best_k = 1
best_acc = 0
k_values = list(range(1, 31))
accs = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_val, knn.predict(X_val))
    accs.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_k = k

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)

print(f"Best k: {best_k}")
print(f"Validation Accuracy: {best_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

plt.figure(figsize=(10, 5))
plt.plot(k_values, accs, marker="o")
plt.axvline(x=best_k, color="gray", linestyle="--", alpha=0.5)
plt.xticks(range(1, 31))
plt.xlabel("k (number of neighbors)")
plt.ylabel("Accuracy")
plt.title("Effect of k on KNN Performance (Iris)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("iris_knn_k_analysis.png", dpi=150)
print("\nPlot saved to iris_knn_k_analysis.png")
