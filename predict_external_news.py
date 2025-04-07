import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

# Load vectorizer
vectorizer = joblib.load("saved_models/vectorizer.jb")

# Load all saved models
models = {
    "Logistic Regression": joblib.load("saved_models/logistic_regression.jb"),
    "Random Forest": joblib.load("saved_models/random_forest.jb"),
    "SVM": joblib.load("saved_models/svm.jb"),
    "Naive Bayes": joblib.load("saved_models/naive_bayes.jb"),
    "Voting Classifier": joblib.load("saved_models/voting_classifier.jb")
}

def preprocess_text(text):
    ps = PorterStemmer()
    stopwords_set = set(stopwords.words("english"))
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords_set]
    return " ".join(text)

def predict_news(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    results = {}
    for name, model in models.items():
        pred = model.predict(vectorized_text)[0]
        proba = getattr(model, "predict_proba", lambda x: [[None, None]])(vectorized_text)
        confidence = max(proba[0]) if proba[0][0] is not None else "N/A"
        results[name] = {"label": "Fake News" if pred == 0 else "Real News", "confidence": confidence}

    return results

def get_majority_vote(results):
    votes = [res["label"] for name, res in results.items() if name != "Voting Classifier"]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]

if __name__ == "__main__":
    input_news = input("Paste the news article text: ")
    results = predict_news(input_news)

    print("\n--- Individual Model Predictions ---")
    for model_name, result in results.items():
        print(f"{model_name}: {result['label']} (Confidence: {result['confidence']})")

    print("\n✅ Final Decision (Voting Classifier):", results["Voting Classifier"]["label"])
    print("📊 Manual Majority Vote:", get_majority_vote(results))
