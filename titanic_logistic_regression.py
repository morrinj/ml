import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay

# ===== TITANIC: sigmoid curve for one feature (Fare) =====
df = pd.read_csv("train.csv")
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
df = df.fillna(df.median(numeric_only=True))

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(random_state=42)
lr.fit(X_scaled, y)

# Titanic: sigmoid curves for Fare and Sex
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for idx, feat_name in enumerate(["Fare", "Sex", "Age"]):
    X_feat = X[[feat_name]].copy()
    ss = StandardScaler()
    X_sc = ss.fit_transform(X_feat)

    lr_feat = LogisticRegression(random_state=42)
    lr_feat.fit(X_sc, y)

    x_range = np.linspace(X_sc.min(), X_sc.max(), 300).reshape(-1, 1)
    probs = lr_feat.predict_proba(x_range)[:, 1]

    ax = axes[idx]
    ax.scatter(X_sc, y, alpha=0.3, label="Data")
    ax.plot(x_range, probs, color="red", linewidth=2, label="Sigmoid (P=Survived)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold (0.5)")
    ax.set_xlabel(f"{feat_name} (scaled)")
    ax.set_ylabel("Probability of Survival")
    ax.set_title(f"Titanic: Logistic Regression ({feat_name})")
    ax.legend()
    ax.grid(alpha=0.3)

# --- Iris: decision boundaries (PetalLength vs PetalWidth) ---
iris = pd.read_csv("iris.csv")
X_iris = iris[["PetalLengthCm", "PetalWidthCm"]]
y_iris = iris["Species"]

scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

lr_iris = LogisticRegression(random_state=42, max_iter=200)
lr_iris.fit(X_iris_scaled, y_iris)

ax2 = axes[3]
DecisionBoundaryDisplay.from_estimator(
    lr_iris, X_iris_scaled,
    response_method="predict",
    alpha=0.4,
    ax=ax2,
    cmap="Pastel1",
)
ax2.scatter(X_iris_scaled[:, 0], X_iris_scaled[:, 1],
            c=pd.factorize(y_iris)[0], cmap="Set1", edgecolors="k")
ax2.set_xlabel("PetalLength (scaled)")
ax2.set_ylabel("PetalWidth (scaled)")
ax2.set_title("Iris: Logistic Regression Boundaries")

plt.tight_layout()
plt.savefig("logistic_regression_diagram.png", dpi=150)
print("Saved to logistic_regression_diagram.png")
