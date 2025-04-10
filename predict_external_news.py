# predict_external_news.py

import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources if not already available
nltk.download("stopwords")
nltk.download("wordnet")

# Load vectorizer
vectorizer = joblib.load("saved_models/tfidf_vectorizer.jb")

# Load all saved models
models = {
    "Logistic Regression": joblib.load("saved_models/logistic_regression.jb"),
    "Random Forest": joblib.load("saved_models/random_forest.jb"),
    "SVM": joblib.load("saved_models/svm.jb"),
    "Naive Bayes": joblib.load("saved_models/naive_bayes.jb"),
    "Voting Classifier": joblib.load("saved_models/voting_classifier.jb")
}

# Preprocessing
lemmatizer = WordNetLemmatizer()
stopwords_set = set(stopwords.words("english"))

def preprocess_text(text):
    """Clean and lemmatize the input text."""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    words = text.split()
    cleaned = [lemmatizer.lemmatize(word) for word in words if word not in stopwords_set]
    return " ".join(cleaned)

def predict_news(text):
    """Predict whether the news article is real or fake using multiple models."""
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    results = {}
    for name, model in models.items():
        pred = model.predict(vectorized_text)[0]
        try:
            proba = model.predict_proba(vectorized_text)
            confidence = max(proba[0])
        except AttributeError:
            confidence = "N/A"

        label = "Fake News" if pred == 0 else "Real News"
        results[name] = {
            "label": label,
            "confidence": confidence
        }

    return results

def get_majority_vote(results):
    """Determine majority prediction with confidence weighting."""
    weighted_votes = {"Fake News": 0, "Real News": 0}

    for name, res in results.items():
        if name != "Voting Classifier":
            label = res["label"]
            confidence = res["confidence"]
            if isinstance(confidence, float):
                weighted_votes[label] += confidence

    return max(weighted_votes, key=weighted_votes.get)
