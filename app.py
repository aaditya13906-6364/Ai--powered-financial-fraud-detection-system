import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# Project Folder
# ==========================================

BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# Page Settings
# ==========================================

st.set_page_config(
    page_title="Financial Fraud Detector",
    page_icon="💳",
    layout="centered"
)


# ==========================================
# Load Model Files
# ==========================================

@st.cache_resource
def load_files():

    # Load trained model
    with open(BASE_DIR / "model.pkl", "rb") as file:
        model = pickle.load(file)

    # Load Amount scaler
    with open(BASE_DIR / "amount_scaler.pkl", "rb") as file:
        amount_scaler = pickle.load(file)

    # Load Time scaler
    with open(BASE_DIR / "time_scaler.pkl", "rb") as file:
        time_scaler = pickle.load(file)

    # Load column names
    with open(BASE_DIR / "columns.pkl", "rb") as file:
        columns = pickle.load(file)

    # Load class counts
    with open(BASE_DIR / "class_counts.pkl", "rb") as file:
        class_counts = pickle.load(file)

    return (
        model,
        amount_scaler,
        time_scaler,
        columns,
        class_counts
    )


# ==========================================
# Check Required Files
# ==========================================

required_files = [
    "model.pkl",
    "amount_scaler.pkl",
    "time_scaler.pkl",
    "columns.pkl",
    "class_counts.pkl"
]

missing_files = []

for file_name in required_files:

    if not (BASE_DIR / file_name).exists():
        missing_files.append(file_name)


if missing_files:

    st.error("⚠️ Required model files are missing.")

    st.write("Missing files:")

    for file_name in missing_files:
        st.write(f"- `{file_name}`")

    st.info(
        "Open the VS Code terminal and run: "
    )

    st.code("python3 model.py")

    st.stop()


# ==========================================
# Load Files
# ==========================================

model, amount_scaler, time_scaler, columns, class_counts = load_files()


# ==========================================
# Title
# ==========================================

st.title("💳 Financial Fraud Detector")

st.write(
    "Enter the transaction details to check whether "
    "the transaction is legitimate or potentially fraudulent."
)

st.divider()


# ==========================================
# Input Section
# ==========================================

st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)


with col1:

    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=100.0,
        step=1.0
    )


with col2:

    time_val = st.number_input(
        "Transaction Time (seconds)",
        min_value=0.0,
        value=40000.0,
        step=100.0
    )


st.caption(
    "Note: This simplified interface uses Amount and Time. "
    "Other transaction features are set to their neutral value."
)


# ==========================================
# Prediction
# ==========================================

if st.button("🔍 Detect Fraud", use_container_width=True):

    # --------------------------------------
    # Scale Amount
    # --------------------------------------

    scaled_amount = amount_scaler.transform(
        np.array([[amount]])
    )[0][0]

    # --------------------------------------
    # Scale Time
    # --------------------------------------

    scaled_time = time_scaler.transform(
        np.array([[time_val]])
    )[0][0]

    # --------------------------------------
    # Create Input Row
    # --------------------------------------

    row = []

    for column in columns:

        if column == "Amount":

            row.append(scaled_amount)

        elif column == "Time":

            row.append(scaled_time)

        else:

            # V1 - V28
            row.append(0.0)

    input_data = np.array(row).reshape(1, -1)

    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    fraud_probability = probabilities[1]

    legitimate_probability = probabilities[0]

    # --------------------------------------
    # Result
    # --------------------------------------

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("🚨 FRAUD TRANSACTION")

        st.write(
            f"**Fraud Probability:** "
            f"{fraud_probability * 100:.2f}%"
        )

        st.write(
            f"**Legitimate Probability:** "
            f"{legitimate_probability * 100:.2f}%"
        )

        st.warning(
            "This transaction has been identified as "
            "potentially fraudulent."
        )

    else:

        st.success("✅ LEGITIMATE TRANSACTION")

        st.write(
            f"**Fraud Probability:** "
            f"{fraud_probability * 100:.2f}%"
        )

        st.write(
            f"**Legitimate Probability:** "
            f"{legitimate_probability * 100:.2f}%"
        )

        st.info(
            "The transaction appears to be legitimate."
        )


# ==========================================
# Dataset Information
# ==========================================

st.divider()

with st.expander("📊 About Dataset"):

    legitimate_count = class_counts.get(0, 0)
    fraud_count = class_counts.get(1, 0)

    total_transactions = (
        legitimate_count + fraud_count
    )

    st.write(
        f"**Total Transactions:** "
        f"{total_transactions:,}"
    )

    st.write(
        f"**Legitimate Transactions:** "
        f"{legitimate_count:,}"
    )

    st.write(
        f"**Fraud Transactions:** "
        f"{fraud_count:,}"
    )

    # --------------------------------------
    # Chart
    # --------------------------------------

    fig, ax = plt.subplots()

    ax.bar(
        ["Legitimate", "Fraud"],
        [
            legitimate_count,
            fraud_count
        ]
    )

    ax.set_ylabel("Number of Transactions")
    ax.set_title("Transaction Distribution")

    st.pyplot(fig)


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "Machine Learning Project | Random Forest | Streamlit"
)