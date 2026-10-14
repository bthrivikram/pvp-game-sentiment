import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. LOAD CLEANED DATA
# ============================================================

train = pd.read_csv("data/train_clean.csv")
validation = pd.read_csv("data/validation_clean.csv")
test = pd.read_csv("data/test_clean.csv")

print("Dataset sizes:")
print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)


# ============================================================
# 3. SEPARATE TEXT AND LABEL
# ============================================================

X_train = train["message"].astype(str).values
y_train = train["target"].values

X_validation = validation["message"].astype(str).values
y_validation = validation["target"].values

X_test = test["message"].astype(str).values
y_test = test["target"].values


# ============================================================
# 4. TOKENIZATION
# ============================================================

MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 50

tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>"
)

# IMPORTANT:
# Fit tokenizer ONLY on training data
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_validation_seq = tokenizer.texts_to_sequences(X_validation)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# ============================================================
# 5. PADDING
# ============================================================

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

X_validation_pad = pad_sequences(
    X_validation_seq,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

vocab_size = min(
    MAX_VOCAB_SIZE,
    len(tokenizer.word_index) + 1
)

print("\nVocabulary size:", vocab_size)

print("Sequence shapes:")
print("Train:", X_train_pad.shape)
print("Validation:", X_validation_pad.shape)
print("Test:", X_test_pad.shape)


# ============================================================
# 6. BUILD 1D CNN
# ============================================================

EMBEDDING_DIM = 100

model = Sequential([
    
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH
    ),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    GlobalMaxPooling1D(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        3,
        activation="softmax"
    )
])


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel Summary:")
model.summary()


# ============================================================
# 8. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ============================================================
# 9. TRAIN CNN
# ============================================================

print("\n==============================")
print("TRAINING CNN")
print("==============================")

history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(
        X_validation_pad,
        y_validation
    ),
    epochs=15,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# 10. VALIDATION RESULTS
# ============================================================

validation_probabilities = model.predict(
    X_validation_pad,
    verbose=0
)

validation_predictions = np.argmax(
    validation_probabilities,
    axis=1
)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

print("\n==============================")
print("VALIDATION RESULTS")
print("==============================")

print("Accuracy:", validation_accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        validation_predictions,
        target_names=[
            "Non-toxic",
            "Mild toxicity",
            "Toxic"
        ]
    )
)


# ============================================================
# 11. TEST RESULTS
# ============================================================

test_probabilities = model.predict(
    X_test_pad,
    verbose=0
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print("Accuracy:", test_accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Non-toxic",
            "Mild toxicity",
            "Toxic"
        ]
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nConfusion Matrix:")
print(cm)