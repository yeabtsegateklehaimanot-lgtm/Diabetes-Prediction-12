# =====================================================
# Diabetes Prediction + Evaluation Pipeline
# Pandas 2.3.3 | Naive Bayes | patient_id alignment
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# =====================================================
# PHASE 1: TRAIN + PREDICT
# =====================================================

Tk().withdraw()

print("Select TRAINING dataset (must include patient_id & diabetic)")
train_file = askopenfilename(filetypes=[("CSV files", "*.csv")])

print("Select TEST dataset (must include patient_id)")
test_file = askopenfilename(filetypes=[("CSV files", "*.csv")])

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

train_data.columns = train_data.columns.str.strip().str.lower()
test_data.columns = test_data.columns.str.strip().str.lower()

if "diabetic" not in train_data.columns:
    raise ValueError("Training data must contain 'diabetic' column")

if "patient_id" not in train_data.columns or "patient_id" not in test_data.columns:
    raise ValueError("Both datasets must contain 'patient_id'")

X_train = train_data.drop(columns=["diabetic"])
y_train = train_data["diabetic"].astype(int)

X_test = test_data[X_train.columns].copy()

# -----------------------------
# Handle missing values
# -----------------------------
for col in X_train.columns:
    med = X_train[col].median()
    X_train[col] = X_train[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

# -----------------------------
# Encode categorical features
# -----------------------------
categorical_cols = [
    "gender", "family_diabetes", "hypertensive",
    "family_hypertension", "cardiovascular_disease", "stroke"
]

for col in categorical_cols:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# -----------------------------
# Feature engineering
# -----------------------------
X_train["bmi_category"] = pd.cut(X_train["bmi"], [0, 25, 30, np.inf], labels=[0,1,2])
X_test["bmi_category"] = pd.cut(X_test["bmi"], [0, 25, 30, np.inf], labels=[0,1,2])

X_train["age_bmi"] = X_train["age"] * X_train["bmi"]
X_test["age_bmi"] = X_test["age"] * X_test["bmi"]

X_train["pulse_pressure"] = X_train["systolic_bp"] - X_train["diastolic_bp"]
X_test["pulse_pressure"] = X_test["systolic_bp"] - X_test["diastolic_bp"]

X_train["high_glucose"] = (X_train["glucose"] > 125).astype(int)
X_test["high_glucose"] = (X_test["glucose"] > 125).astype(int)

# -----------------------------
# Transform + SMOTE
# -----------------------------
pt = PowerTransformer(method="yeo-johnson", standardize=True)
X_train = pt.fit_transform(X_train)
X_test = pt.transform(X_test)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# -----------------------------
# Feature selection
# -----------------------------
selector = SelectKBest(f_classif, k="all")
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
model = GaussianNB(var_smoothing=1e-8)
model.fit(X_train, y_train)

# -----------------------------
# Predict
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

predictions = test_data.copy()
predictions["predicted_class"] = y_pred
predictions["y_prob"] = y_prob

predictions.to_csv("diabetes_predictions.csv", index=False)
print("\nPredictions saved as diabetes_predictions.csv")

# =====================================================
# PHASE 2: EVALUATION USING SEPARATE FILES
# =====================================================

print("\nSelect TRUE LABELS CSV (patient_id, diabetic)")
true_file = askopenfilename(filetypes=[("CSV files", "*.csv")])

print("Select PREDICTED CSV (patient_id, predicted_class, y_prob)")
pred_file = askopenfilename(filetypes=[("CSV files", "*.csv")])

true_df = pd.read_csv(true_file)
pred_df = pd.read_csv(pred_file)

data = pd.merge(
    true_df, pred_df,
    on="patient_id",
    how="inner"
)

print(f"\nMatched rows used for evaluation: {len(data)}")

y_true = data["diabetic"].astype(int)
y_pred = data["predicted_class"].astype(int)
y_prob = data["y_prob"].astype(float)

# -----------------------------
# Metrics
# -----------------------------
print("\n----- MODEL EVALUATION -----")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
print("F1-score :", f1_score(y_true, y_pred, zero_division=0))
print("\nClassification Report:\n",
      classification_report(y_true, y_pred, zero_division=0))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Precision–Recall Curve
# -----------------------------
precision, recall, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid()
plt.show()

