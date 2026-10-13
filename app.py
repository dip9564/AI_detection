import streamlit as st
import pickle as pkl
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from linguistic_features import extract_linguistic_features
from scipy.sparse import hstack, csr_matrix

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def get_model():
    with open(BASE_DIR / "final_model.pkl", "rb") as f:
        model = pkl.load(f)

    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pkl.load(f)

    with open(BASE_DIR / "vectorzier.pkl", "rb") as f:
        vectorizer = pkl.load(f)

    with open(BASE_DIR / "semantic_scaler.pkl", "rb") as f:
        semantic_scaler = pkl.load(f)

    semantic_model = SentenceTransformer(
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
        device="cpu"
    )

    return {
        "model": model,
        "scaler": scaler,
        "vectorizer": vectorizer,
        "semantic_scaler": semantic_scaler,
        "semantic_model": semantic_model,
    }


def predict(text: str):
    artifacts = get_model()

    model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]
    scaler = artifacts["scaler"]
    semantic_scaler = artifacts["semantic_scaler"]
    semantic_model = artifacts["semantic_model"]

    # 1. TF-IDF features
    features_tfidf = vectorizer.transform([text])

    # 2. Linguistic features
    linguistic = np.asarray(
        [extract_linguistic_features(text)],
        dtype=np.float32
    )
    features_linguistic = scaler.transform(linguistic)

    # 3. Semantic embeddings
    semantic = semantic_model.encode(
        [text],
        batch_size=1,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    features_semantic = semantic_scaler.transform(semantic)

    # 4. Combine in exactly the same order as training
    X_combined = hstack([
        features_tfidf,
        csr_matrix(features_linguistic),
        csr_matrix(features_semantic)
    ]).tocsr()

    prediction = int(model.predict(X_combined)[0])
    probabilities = model.predict_proba(X_combined)[0]

    return {
        "label": "human" if prediction == 0 else "ai",
        "human_probability": float(probabilities[0]),
        "ai_probability": float(probabilities[1]),
    }


st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="wide"
)

st.title("AI Text Detector")
st.write("Enter text to send to the AI detection service.")

text = st.text_area(
    "Text to analyze",
    height=220,
    placeholder="Paste text here..."
)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            result = predict(text)

            st.subheader("Result")
            is_ai = result["label"] == "ai"

            if is_ai:
                st.success("AI detected")
            else:
                st.success("Human-written text")

            col1, col2 = st.columns(2)
            col1.metric(
                "Human-written",
                f'{result["human_probability"] * 100:.2f}%'
            )
            col2.metric(
                "AI-written",
                f'{result["ai_probability"] * 100:.2f}%'
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
