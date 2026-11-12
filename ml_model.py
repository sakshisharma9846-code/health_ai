"""Machine-learning symptom matcher.

Pipeline: TF-IDF (word 1-2 grams) -> Logistic Regression.
Training data is generated from the keyword lists in data/problems.json.
"""
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

PREFIX = ["", "", "i have ", "i am suffering from ", "i feel ", "there is ", "my "]
SUFFIX = ["", "", " since morning", " for two days", " today", " since yesterday"]


def make_dataset(problems, per_class=60, seed=42):
    rng = random.Random(seed)
    X, y = [], []
    for p in problems:
        kws = p["keywords"]
        for _ in range(per_class):
            k = rng.randint(1, min(3, len(kws)))
            text = " and ".join(rng.sample(kws, k))
            X.append(rng.choice(PREFIX) + text + rng.choice(SUFFIX))
            y.append(p["id"])
    return X, y


class SymptomModel:
    def __init__(self, problems):
        X, y = make_dataset(problems)
        self.pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=20),
        )
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
        self.pipe.fit(Xtr, ytr)
        self.accuracy = self.pipe.score(Xte, yte)  # accuracy on held-out synthetic data
        self.pipe.fit(X, y)

    def predict(self, text, top=3):
        probs = self.pipe.predict_proba([text.lower()])[0]
        classes = self.pipe.classes_
        order = sorted(range(len(classes)), key=lambda i: -probs[i])[:top]
        return [(int(classes[i]), float(probs[i])) for i in order]
