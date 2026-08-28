import pandas as pd
import streamlit as st
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

fake_path = DATA_DIR / "fake.csv"
true_path = DATA_DIR / "true.csv"

if not fake_path.exists() or not true_path.exists():
    raise FileNotFoundError(
        "Put fake.csv and true.csv inside the data folder before running this script."
    )

fake = pd.read_csv(fake_path,encoding="latin1", on_bad_lines="skip",engine="python")
true = pd.read_csv(true_path,encoding="latin1",on_bad_lines="skip",engine="python")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")

data = data[["text", "label"]].dropna()

X = data["text"]
Y = data["label"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

with open(BASE_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(BASE_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nmodel.pkl and vectorizer.pkl created successfully.")
