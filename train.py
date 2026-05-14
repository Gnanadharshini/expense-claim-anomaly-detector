import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("data.csv")

print("Original Dataset:")
print(df)


# Feature 1: Amount per frequency
df['amount_per_frequency'] = df['amount'] / df['frequency']

# Feature 2: High amount flag
df['high_amount'] = df['amount'].apply(
    lambda x: 1 if x > 10000 else 0
)

# Feature 3: Encode category
encoder = LabelEncoder()
df['category_encoded'] = encoder.fit_transform(df['category'])

print("\nDataset After Feature Extraction:")
print(df)



X = df[[
    'amount',
    'frequency',
    'amount_per_frequency',
    'high_amount',
    'category_encoded'
]]



model = IsolationForest(contamination=0.3)
model.fit(X)


df['prediction'] = model.predict(X)

df['result'] = df['prediction'].apply(
    lambda x: "Anomaly" if x == -1 else "Normal"
)

print("\nPrediction Results:")
print(df)


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\nModel and encoder saved!")