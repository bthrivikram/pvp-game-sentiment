import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib
import os

# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

train = pd.read_csv("data/train_clean.csv")
validation = pd.read_csv("data/validation_clean.csv")
test = pd.read_csv("data/test_clean.csv")

print("Dataset sizes:")
print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)


# ============================================================
# 2. SEPARATE TEXT AND LABEL
# ============================================================

X_train = train["message"]
y_train = train["target"]

X_validation = validation["message"]
y_validation = validation["target"]

X_test = test["message"]
y_test = test["target"]


# ============================================================
# 3. TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

# IMPORTANT:
# Fit TF-IDF ONLY on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Only transform validation/test
X_validation_tfidf = vectorizer.transform(X_validation)
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF shapes:")
print("Train:", X_train_tfidf.shape)
print("Validation:", X_validation_tfidf.shape)
print("Test:", X_test_tfidf.shape)


# ============================================================
# 4. TRAIN LINEAR SVM
# ============================================================

model = LinearSVC()

model.fit(X_train_tfidf, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/svm_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nSVM model and TF-IDF vectorizer saved.")


# ============================================================
# 5. VALIDATION RESULTS
# ============================================================

validation_predictions = model.predict(X_validation_tfidf)

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
# 6. TEST RESULTS
# ============================================================

test_predictions = model.predict(X_test_tfidf)

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
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nConfusion Matrix:")
print(cm)