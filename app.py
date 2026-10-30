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
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1 {
        font-weight: 700;
        letter-spacing: -0.04em;
        margin-bottom: 0.25rem;
    }

    [data-testid="stTextArea"] textarea {
        border-radius: 12px;
        line-height: 1.6;
        min-height: 190px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }

    .stButton > button {
        border-radius: 11px;
        font-weight: 600;
        min-height: 2.8rem;
        transition: transform 120ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.write("**CUSTOMER CARE · SMART CLASSIFICATION**")
st.title("💬 Support, sorted.")
st.caption("Turn a customer message into the right support category in seconds.")

model = load_or_train_model()

overview_cols = st.columns(3)
overview_cols[0].metric("Support categories", len(model.classes_))
overview_cols[1].metric("Input needed", "One message")
overview_cols[2].metric("Results", "Top matches")

st.write("")
input_col, guide_col = st.columns([1.7, 1], gap="large")

with input_col:
    with st.container(border=True):
        st.subheader("Classify a message")
        st.caption("Paste or type the message you received from a customer.")
        text_input = st.text_area(
            "Customer message",
            height=180,
            placeholder="Example: I still haven't received my bank card after two weeks."
        )
        predict_clicked = st.button("✨  Predict category", type="primary", width="stretch")

with guide_col:
    with st.container(border=True):
        st.subheader("Get a better match")
        st.caption("A little context helps the classifier find the right team.")
        st.markdown(
            """
            **Include details like**
            - What product or service is involved
            - What happened and when
            - What help the customer needs

            **Example**

            > My card has not arrived after two weeks. Can you check the delivery?
            """
        )

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
            st.subheader("Your prediction")
            st.caption("The closest matching support category and model confidence.")
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
    with st.expander("Explore sample training messages"):
        st.dataframe(train_df.head(10), width="stretch")
