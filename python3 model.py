import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# 1. Load the dataset
data = pd.read_csv("creditcard.csv")
print("Total transactions:", len(data))
print(data["Class"].value_counts())

# 2. Scale Amount and Time (they have big numbers compared to other columns)
scaler = StandardScaler()
data["Amount"] = scaler.fit_transform(data[["Amount"]])
data["Time"] = scaler.fit_transform(data[["Time"]])

# 3. Split into inputs (X) and output (y)
X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train the model
# class_weight="balanced" helps because fraud cases are very rare (only ~0.17%)
model = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# 5. Check how well it performs
predictions = model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall   :", recall_score(y_test, predictions))
print("F1-score :", f1_score(y_test, predictions))

# 6. Save the model and scaler so app.py can use them later
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# Save class counts too, so the web app can show a chart without
# having to load the full (large) CSV file again.
class_counts = data["Class"].value_counts().to_dict()
with open("class_counts.pkl", "wb") as f:
    pickle.dump(class_counts, f)

print("Done! Model saved as model.pkl")