import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ============================================================
# TITANIC
# ============================================================
print("=" * 60)
print("TITANIC - Logistic Regression")
print("=" * 60)

df_t = pd.read_csv("train.csv")
df_t["Sex"] = df_t["Sex"].map({"male": 0, "female": 1})
df_t["Embarked"] = df_t["Embarked"].map({"S": 0, "C": 1, "Q": 2})
df_t = df_t.fillna(df_t.median(numeric_only=True))

features_t = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X_t = df_t[features_t]
y_t = df_t["Survived"]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.2, random_state=42, stratify=y_t)

scaler_t = StandardScaler()
X_train_t = scaler_t.fit_transform(X_train_t)
X_test_t = scaler_t.transform(X_test_t)

model_t = LogisticRegression(max_iter=1000, random_state=42)
model_t.fit(X_train_t, y_train_t)
y_pred_t = model_t.predict(X_test_t)

cm_t = confusion_matrix(y_test_t, y_pred_t)
print("\nConfusion Matrix:")
print(cm_t)
print(f"\nAccuracy: {accuracy_score(y_test_t, y_pred_t):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_t, y_pred_t, target_names=["Not Survived", "Survived"]))

plt.figure(figsize=(6, 5))
classes_t = ['Not Survived', 'Survived']
plt.imshow(cm_t, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Titanic Logistic Regression')
plt.colorbar()
tick_marks_t = np.arange(len(classes_t))
plt.xticks(tick_marks_t, classes_t)
plt.yticks(tick_marks_t, classes_t)
thresh_t = cm_t.max() / 2.
for i in range(cm_t.shape[0]):
    for j in range(cm_t.shape[1]):
        plt.text(j, i, format(cm_t[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm_t[i, j] > thresh_t else "black")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_titanic.png')
print("\nSaved: confusion_matrix_titanic.png")

# ============================================================
# IRIS
# ============================================================
print("\n" + "=" * 60)
print("IRIS - Logistic Regression")
print("=" * 60)

df_i = pd.read_csv("iris.csv")
features_i = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X_i = df_i[features_i]
y_i = df_i["Species"]

le_i = LabelEncoder()
y_i_enc = le_i.fit_transform(y_i)

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_i, y_i_enc, test_size=0.2, random_state=42, stratify=y_i_enc)

scaler_i = StandardScaler()
X_train_i = scaler_i.fit_transform(X_train_i)
X_test_i = scaler_i.transform(X_test_i)

model_i = LogisticRegression(max_iter=1000, random_state=42)
model_i.fit(X_train_i, y_train_i)
y_pred_i = model_i.predict(X_test_i)

cm_i = confusion_matrix(y_test_i, y_pred_i)
print("\nConfusion Matrix:")
print(cm_i)
print(f"\nAccuracy: {accuracy_score(y_test_i, y_pred_i):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_i, y_pred_i, target_names=le_i.classes_))

plt.figure(figsize=(6, 5))
classes_i = le_i.classes_
plt.imshow(cm_i, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Iris Logistic Regression')
plt.colorbar()
tick_marks_i = np.arange(len(classes_i))
plt.xticks(tick_marks_i, classes_i, rotation=45)
plt.yticks(tick_marks_i, classes_i)
thresh_i = cm_i.max() / 2.
for i in range(cm_i.shape[0]):
    for j in range(cm_i.shape[1]):
        plt.text(j, i, format(cm_i[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm_i[i, j] > thresh_i else "black")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_iris.png')
print("\nSaved: confusion_matrix_iris.png")
