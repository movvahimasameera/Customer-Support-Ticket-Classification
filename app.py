import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "customer_support_classifier.pkl"
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    train_df = pd.read_csv(TRAIN_PATH)
    X_train = train_df["text"].apply(clean_text)
    y_train = train_df["category"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)
    return model


def load_or_train_model():
    MODEL_DIR.mkdir(exist_ok=True)

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    model = build_model()
    joblib.dump(model, MODEL_PATH)
    return model


st.set_page_config(page_title="Customer Support Classifier", page_icon="💬", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-weight: 700;
        letter-spacing: -0.035em;
    }

    [data-testid="stTextArea"] textarea {
        border-radius: 12px;
        line-height: 1.6;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 1rem;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.1rem;
        transition: transform 120ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("💬 Customer Support Classifier")
st.caption("Turn a customer message into a clear support category.")

model = load_or_train_model()

with st.container(border=True):
    st.subheader("Try a message")
    text_input = st.text_area(
        "Enter a customer support message",
        height=150,
        placeholder="Example: I still haven't received my bank card after two weeks."
    )

    predict_clicked = st.button("Predict category", type="primary")

if predict_clicked:
    if not text_input.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(text_input)
        pred = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0]
        class_index = list(model.classes_).index(pred)
        confidence = probabilities[class_index] * 100

        top_classes = sorted(
            zip(model.classes_, probabilities),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        with st.container(border=True):
            st.subheader("Prediction")
            category_col, confidence_col = st.columns(2)
            category_col.metric("Support category", pred)
            confidence_col.metric("Confidence", f"{confidence:.2f}%")

            with st.expander("Explore confidence by category"):
                for label, probability in top_classes:
                    st.caption(f"{label} · {probability:.1%}")
                    st.progress(float(probability))

st.divider()

if TRAIN_PATH.exists():
    train_df = pd.read_csv(TRAIN_PATH)
    with st.expander("Preview training examples"):
        st.dataframe(train_df.head(10), width="stretch")
