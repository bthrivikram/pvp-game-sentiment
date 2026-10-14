
import json
import re
import string

import joblib
import numpy as np
import onnxruntime as ort
import streamlit as st


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PvP Game Chat Classifier",
    page_icon="🎮",
    layout="wide"
)


# -----------------------------
# Load all four models
# -----------------------------
@st.cache_resource
def load_models():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    svm_model = joblib.load("models/svm_model.pkl")
    rf_model = joblib.load("models/rf_model.pkl")
    xgb_model = joblib.load("models/xgboost_model.pkl")

    cnn_session = ort.InferenceSession(
        "models/cnn_model.onnx",
        providers=["CPUExecutionProvider"]
    )

    with open("models/cnn_tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    return (
        vectorizer,
        svm_model,
        rf_model,
        xgb_model,
        cnn_session,
        tokenizer
    )


(
    vectorizer,
    svm_model,
    rf_model,
    xgb_model,
    cnn_session,
    tokenizer
) = load_models()


# -----------------------------
# Label names
# -----------------------------
label_names = {
    0: "Non-toxic",
    1: "Mild toxicity",
    2: "Toxic"
}


# -----------------------------
# CNN text preprocessing
# -----------------------------
def cnn_preprocess(text, tokenizer, max_length=50):
    word_index = tokenizer["word_index"]
    filters = tokenizer["filters"]
    lower = tokenizer["lower"]
    split_char = tokenizer["split"]
    oov_token = tokenizer["oov_token"]
    num_words = tokenizer["num_words"]

    if lower:
        text = text.lower()

    # Match Keras Tokenizer's punctuation filtering
    translation = str.maketrans({
        char: split_char for char in filters
    })
    words = text.translate(translation).split(split_char)

    sequence = []
    oov_index = word_index.get(oov_token) if oov_token else None

    for word in words:
        if not word:
            continue

        index = word_index.get(word)

        if index is None:
            if oov_index is not None:
                sequence.append(oov_index)
        elif num_words and index >= num_words:
            if oov_index is not None:
                sequence.append(oov_index)
        else:
            sequence.append(index)

    # Keras pad_sequences defaults: pre-padding and pre-truncation
    sequence = sequence[-max_length:]
    padded = [0] * (max_length - len(sequence)) + sequence

    return np.asarray([padded], dtype=np.float32)


def predict_cnn(text):
    cnn_input = cnn_preprocess(text, tokenizer)

    input_name = cnn_session.get_inputs()[0].name
    outputs = cnn_session.run(
        None,
        {input_name: cnn_input}
    )

    probabilities = np.asarray(outputs[0])[0]
    return int(np.argmax(probabilities))


# -----------------------------
# User interface
# -----------------------------
st.title("🎮 PvP Game Chat Toxicity Classifier")

st.write(
    "Enter a PvP game chat message and compare predictions "
    "from four machine learning models."
)

st.caption(
    "Non-toxic: ordinary or non-abusive chat | "
    "Mild toxicity: borderline insults or mildly abusive chat | "
    "Toxic: strongly abusive or offensive chat"
)

message = st.text_area(
    "Enter a game chat message:",
    placeholder="Example: you're playing really well!",
    height=120
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message.")

    else:
        try:
            # Classical models use TF-IDF
            message_tfidf = vectorizer.transform([message])

            predictions = {
                "Linear SVM": int(
                    svm_model.predict(message_tfidf)[0]
                ),
                "Random Forest": int(
                    rf_model.predict(message_tfidf)[0]
                ),
                "XGBoost": int(
                    xgb_model.predict(message_tfidf)[0]
                ),
                "1D CNN": predict_cnn(message)
            }

            st.subheader("Model Predictions")

            # Agreement indicator
            if len(set(predictions.values())) == 1:
                st.success(
                    "All four models agree on the prediction."
                )
            else:
                st.info(
                    "The models disagree. Review the individual "
                    "predictions; agreement does not guarantee correctness."
                )

            # Display all four models
            columns = st.columns(4)

            for column, (model_name, prediction) in zip(
                columns, predictions.items()
            ):
                with column:
                    st.metric(
                        label=model_name,
                        value=label_names[prediction]
                    )

            st.subheader("Detailed Results")

            for model_name, prediction in predictions.items():
                text = f"**{model_name}:** {label_names[prediction]}"

                if prediction == 0:
                    st.success(text)
                elif prediction == 1:
                    st.warning(text)
                else:
                    st.error(text)

        except Exception as error:
            st.error("Prediction failed. Check the model files and error below.")
            st.exception(error)
