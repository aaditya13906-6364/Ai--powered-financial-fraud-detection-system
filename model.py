import pandas as pd
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------------
# Project folder
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

dataset_path = BASE_DIR / "creditcard.csv"

data = pd.read_csv(dataset_path)

print("\n==============================")
print("Financial Fraud Detection")
print("==============================")

print("Total transactions:", len(data))
print("\nClass distribution:")
print(data["Class"].value_counts())

# ---------------------------------------
# 2. Scale Amount and Time separately
# ---------------------------------------

amount_scaler = StandardScaler()
time_scaler = StandardScaler()

data["Amount"] = amount_scaler.fit_transform(
    data[["Amount"]]
)

data["Time"] = time_scaler.fit_transform(
    data[["Time"]]
)

# ---------------------------------------
# 3. Separate input and output
# ---------------------------------------

X = data.drop("Class", axis=1)
y = data["Class"]

# ---------------------------------------
# 4. Train/Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining transactions:", len(X_train))
print("Testing transactions:", len(X_test))

# ---------------------------------------
# 5. Train Random Forest Model
# ---------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

# ---------------------------------------
# 6. Model Evaluation
# ---------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# ---------------------------------------
# 7. Save Model
# ---------------------------------------

model_path = BASE_DIR / "model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

# ---------------------------------------
# 8. Save Amount Scaler
# ---------------------------------------

amount_scaler_path = BASE_DIR / "amount_scaler.pkl"

with open(amount_scaler_path, "wb") as file:
    pickle.dump(amount_scaler, file)

# ---------------------------------------
# 9. Save Time Scaler
# ---------------------------------------

time_scaler_path = BASE_DIR / "time_scaler.pkl"

with open(time_scaler_path, "wb") as file:
    pickle.dump(time_scaler, file)

# ---------------------------------------
# 10. Save Column Names
# ---------------------------------------

columns_path = BASE_DIR / "columns.pkl"

with open(columns_path, "wb") as file:
    pickle.dump(list(X.columns), file)

# ---------------------------------------
# 11. Save Class Counts
# ---------------------------------------

class_counts = data["Class"].value_counts().to_dict()

class_counts_path = BASE_DIR / "class_counts.pkl"

with open(class_counts_path, "wb") as file:
    pickle.dump(class_counts, file)

# ---------------------------------------
# 12. Final Message
# ---------------------------------------

print("\n==============================")
print("FILES CREATED SUCCESSFULLY")
print("==============================")

print("model.pkl")
print("amount_scaler.pkl")
print("time_scaler.pkl")
print("columns.pkl")
print("class_counts.pkl")

print("\nModel is ready for Streamlit!")