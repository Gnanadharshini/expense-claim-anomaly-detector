from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Load encoder
encoder = pickle.load(open("encoder.pkl", "rb"))

# Home route
@app.get("/")
def home():
    return {"message": "Expense Claim API Running"}

# Prediction route
@app.post("/predict")
def predict(amount: float, frequency: int, category: str):

    # -----------------------
    # FEATURE EXTRACTION
    # -----------------------

    # Feature 1
    amount_per_frequency = amount / frequency

    # Feature 2
    high_amount = 1 if amount > 10000 else 0

    # Feature 3
    category_encoded = encoder.transform([category])[0]

    # Final feature array
    data = np.array([[
        amount,
        frequency,
        amount_per_frequency,
        high_amount,
        category_encoded
    ]])

    # ML prediction
    prediction = model.predict(data)

    # Rule-based detection
    if amount > 10000 or frequency > 5:
        result = "Anomaly"
    else:
        result = "Normal"

    # Risk score
    score = model.decision_function(data)[0]

    # Return result
    return {
        "result": result,
        "risk_score": float(score)
    }