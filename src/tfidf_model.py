from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_tfidf_model(X_train, y_train):
    """
    Train TF-IDF + Logistic Regression sentiment classifier.
    """

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    return model, vectorizer


def predict_tfidf(model, vectorizer, X_test):
    """
    Generate predictions and probabilities.
    """

    X_test_vec = vectorizer.transform(X_test)

    predictions = model.predict(X_test_vec)
    probabilities = model.predict_proba(X_test_vec)[:, 1]

    return predictions, probabilities
