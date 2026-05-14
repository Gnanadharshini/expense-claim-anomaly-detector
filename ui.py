import streamlit as st
import requests
import pandas as pd

# -----------------------------
# SESSION STORAGE
# -----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# TITLE
# -----------------------------

st.title("💰 Expense Claim Anomaly Detector")

st.write("AI-based fraud detection system for expense claims")

# -----------------------------
# USER INPUTS
# -----------------------------

amount = st.number_input(
    "Enter Amount",
    min_value=0.0
)

frequency = st.number_input(
    "Enter Frequency",
    min_value=1
)

category = st.selectbox(
    "Select Category",
    ["food", "travel", "shopping", "medical"]
)

# -----------------------------
# BUTTON
# -----------------------------

if st.button("Check Claim"):

    # -----------------------------
    # SEND DATA TO API
    # -----------------------------

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        params={
            "amount": amount,
            "frequency": frequency,
            "category": category
        }
    )

    # -----------------------------
    # GET RESULT
    # -----------------------------

    result = response.json()

    # -----------------------------
    # SHOW PREDICTION
    # -----------------------------

    st.subheader("Prediction Result")

    st.write(result["result"])

    # -----------------------------
    # SHOW RISK SCORE
    # -----------------------------

    st.subheader("Risk Score")

    st.write(result["risk_score"])

    # -----------------------------
    # DASHBOARD
    # -----------------------------

    st.subheader("Dashboard")

    if result["result"] == "Anomaly":
        st.metric("Fraud Status", "High Risk")
    else:
        st.metric("Fraud Status", "Low Risk")

    # -----------------------------
    # EXPLANATION PANEL
    # -----------------------------

    st.subheader("Explanation Panel")

    if amount > 10000:
        st.write("⚠ High expense amount detected")

    if frequency > 5:
        st.write("⚠ High claim frequency detected")

    if category == "travel":
        st.write("⚠ Travel claims monitored for unusual activity")

    # -----------------------------
    # STORE HISTORY
    # -----------------------------

    st.session_state.history.append({
        "Amount": amount,
        "Frequency": frequency,
        "Category": category,
        "Result": result["result"],
        "Risk Score": result["risk_score"]
    })

# -----------------------------
# REVIEW QUEUE
# -----------------------------

st.subheader("Review Queue")

history_df = pd.DataFrame(st.session_state.history)

st.dataframe(history_df)

# -----------------------------
# DOWNLOAD REPORT
# -----------------------------

if not history_df.empty:

    csv = history_df.to_csv(index=False)

    st.download_button(
        label="Download Report",
        data=csv,
        file_name="expense_report.csv",
        mime="text/csv"
    )