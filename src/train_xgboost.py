import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from xgboost import XGBClassifier

# ============================================================
# 1. LOAD TRAINING AND VALIDATION DATA
# ============================================================

train = pd.read_csv("data/train_clean.csv")
validation = pd.read_csv("data/validation_clean.csv")

X_train = train["message"].astype(str)
y_train = train["target"].astype(int)

X_val = validation["message"].astype(str)
y_val = validation["target"].astype(int)

print("Train size:", len(train))
print("Validation size:", len(validation))


# ============================================================
# 2. TF-IDF VECTORIZATION
# Use the original TF-IDF settings
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)


# ============================================================
# 3. DEFINE XGBOOST AND PARAMETERS TO TRY
# ============================================================

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=1,
    tree_method="hist"
)

param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


# ============================================================
# 4. TUNE USING TRAINING DATA ONLY
# ============================================================

print("\nTuning XGBoost. This may take several minutes...")

search.fit(X_train_tfidf, y_train)

best_model = search.best_estimator_

print("\nBest parameters:", search.best_params_)
print("Best cross-validation macro F1:", round(search.best_score_, 4))


# ============================================================
# 5. EVALUATE ON VALIDATION DATA
# ============================================================

val_predictions = best_model.predict(X_val_tfidf)

print("\n==============================")
print("TUNED XGBOOST: VALIDATION")
print("==============================")

print("Accuracy:", round(accuracy_score(y_val, val_predictions), 4))
print("Macro F1:", round(
    f1_score(y_val, val_predictions, average="macro"), 4
))

print("\nClassification Report:")
print(classification_report(
    y_val,
    val_predictions,
    labels=[0, 1, 2],
    target_names=["Non-toxic", "Mild toxicity", "Toxic"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions, labels=[0, 1, 2]))


# ============================================================
# 6. SAVE TUNED MODEL AND MATCHING VECTORIZER
# Keep original files untouched
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/xgboost_model_tuned.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer_xgb_tuned.pkl")

print("\nSaved tuned model: models/xgboost_model_tuned.pkl")
print("Saved matching vectorizer: models/tfidf_vectorizer_xgb_tuned.pkl")
print("Original XGBoost and shared vectorizer files were not overwritten.")
