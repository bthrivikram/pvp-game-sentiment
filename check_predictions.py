
import joblib
import pandas as pd

# Load the original models and TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
svm = joblib.load("models/svm_model.pkl")
rf = joblib.load("models/rf_model.pkl")
xgb = joblib.load("models/xgboost_model.pkl")

# Load real, labelled test examples
df = pd.read_csv("data/test_clean.csv").dropna(subset=["message", "target"])

label_names = {
    0: "Non-toxic",
    1: "Mild toxicity",
    2: "Toxic"
}

# Check whether each model expects the same number of features
print("TF-IDF features:", len(vectorizer.vocabulary_))
print("SVM features:", svm.n_features_in_)
print("RF features:", rf.n_features_in_)
print("XGBoost features:", xgb.n_features_in_)

# Predict on 10 real examples
sample = df.sample(n=min(10, len(df)), random_state=42)
X = vectorizer.transform(sample["message"].astype(str))

predictions = {
    "Actual": sample["target"].astype(int).map(label_names).values,
    "SVM": [label_names[int(x)] for x in svm.predict(X)],
    "Random Forest": [label_names[int(x)] for x in rf.predict(X)],
    "XGBoost": [label_names[int(x)] for x in xgb.predict(X)],
}

results = pd.DataFrame(predictions, index=sample.index)
results.insert(0, "Message", sample["message"].values)

print("\nPredictions on real test examples:\n")
print(results.to_string(index=False))


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

train = pd.read_csv("data/train_clean.csv")
train["message"] = train["message"].fillna("").astype(str)

fresh_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

fresh_vectorizer.fit(train["message"])

saved_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print(
    "Same vocabulary mapping:",
    saved_vectorizer.vocabulary_ == fresh_vectorizer.vocabulary_
)
