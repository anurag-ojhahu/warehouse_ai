import joblib

model = joblib.load("section2_ml/risk_classifier.pkl")

samples = [
    "fragile glass box",
    "handle with care fragile item",
    "flammable chemical liquid",
    "standard cardboard package"
]

for text in samples:
    prediction = model.predict([text])[0]
    print(f"Input: {text}")
    print(f"Predicted Risk: {prediction}")
    print("-" * 40)