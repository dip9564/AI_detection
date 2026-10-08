import __main__
from pathlib import Path
import joblib
import streamlit as st
from model_features import FeatureExtractor


MODEL_PATH = Path(__file__).resolve().parent / "finalized_model.joblib"

# The saved model was created in a notebook, so it references
# __main__.FeatureExtractor during deserialization.
__main__.FeatureExtractor = FeatureExtractor


def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="AI Text Detector",page_icon="🤖",layout="wide")
st.title("AI Text Detector")
st.write("Enter text to estimate whether it was written by AI or a human.")

text = st.text_area("Text to analyze", height=220, placeholder="Paste text here...")

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            model = load_model()
            prediction = int(model.predict([text])[0])
            probability = model.predict_proba([text])[0]

            st.subheader("Result")
            st.success("AI detected" if prediction == 1 else "Human-written text")

            col1, col2 = st.columns(2)
            col1.metric("AI-written", f"{probability[1] * 100 :.2f}%")
            col2.metric("Human-written", f"{probability[0] * 100 :.2f}%")
        except FileNotFoundError:
            st.error(f"Model file not found")
        except Exception as error:
            st.error(f"Unable to analyze the text: {error}")
