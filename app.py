import streamlit as st
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

st.title("📰 Fake News Detection System")
st.write("Enter a news article below and check whether the trained model predicts it as real or fake.")

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error("Model files are missing. Run train_model.py first.")
    st.stop()

news = st.text_area(
    "Enter News Article",
    height=220,
    placeholder="Paste or type the news article here..."
)

if st.button("🔍 Check News", use_container_width=True):
    if not news.strip():
        st.warning("Please enter a news article.")
    else:
        features = vectorizer.transform([news])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        if prediction == 1:
            confidence = probabilities[1] * 100
            st.success(f"🟢 REAL NEWS\n\nConfidence: {confidence:.2f}%")
        else:
            confidence = probabilities[0] * 100
            st.error(f"🔴 FAKE NEWS\n\nConfidence: {confidence:.2f}%")

st.caption("Note: This is a machine-learning classifier trained on the supplied dataset. It is not a live fact-checking service.")
