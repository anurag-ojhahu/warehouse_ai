import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Load dataset
data = pd.read_csv("section2_ml/sample_dataset.csv")

X = data["text"]
y = data["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline (Vectorizer + Model)
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, predictions))

# Save model
with open("section2_ml/risk_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as risk_classifier.pkl")