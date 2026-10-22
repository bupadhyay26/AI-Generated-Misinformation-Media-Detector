"""
train.py  -  (optional) retrains the model. The trained model is already included in model/.

Needs the ISOT dataset:  data/Fake.csv  and  data/True.csv
    (download from https://github.com/Ppius6/Fake-News-Detection-Model  -> Fake.csv, True.csv)
Run:  .venv\\Scripts\\python train.py
"""
import argparse
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from detector import clean_text

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
parser.add_argument("--samples", type=int, default=5000, help="articles per class")
args = parser.parse_args()

# 1. load + sample + clean
fake = pd.read_csv(f"{args.data_dir}/Fake.csv").sample(args.samples, random_state=42)
real = pd.read_csv(f"{args.data_dir}/True.csv").sample(args.samples, random_state=42)
df = pd.concat([fake.assign(label=0), real.assign(label=1)], ignore_index=True)   # 0 = FAKE, 1 = REAL
df["clean"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean_text)
print("articles:", len(df))

# 2. 80/20 split (stratified)
X_train, X_test, y_train, y_test = train_test_split(df["clean"], df["label"], test_size=0.2,
                                                    random_state=42, stratify=df["label"])

# 3. compare three models on the same split
def tfidf():
    return TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced", random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=40, n_jobs=-1, random_state=42),
}
rows, fitted = [], {}
for name, clf in candidates.items():
    pipe = make_pipeline(tfidf(), clf).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    fitted[name] = pipe
    rows.append({"model": name, "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
                 "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred)})
    print(rows[-1])

best = max(rows, key=lambda r: r["f1"])["model"]
tn, fp, fn, tp = confusion_matrix(y_test, fitted[best].predict(X_test)).ravel()

# 4. 5-fold cross-validation of the chosen model (shows the score is not a lucky split)
cv = cross_val_score(make_pipeline(tfidf(), candidates[best]), df["clean"], df["label"],
                     cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")

# 5. save only the chosen model
pipe = fitted[best]
joblib.dump({"vectorizer": pipe.steps[0][1], "model": pipe.steps[1][1]}, "model/model.joblib", compress=3)
json.dump({
    "dataset": "ISOT Fake/True news (Kaggle), sampled", "articles_total": len(df), "n_train": len(X_train),
    "n_test": len(X_test), "best_model": best, "models": rows,
    "confusion": {"fake_correct": int(tn), "fake_called_real": int(fp), "real_called_fake": int(fn), "real_correct": int(tp)},
    "cv_accuracy_mean": float(cv.mean()), "cv_accuracy_std": float(cv.std()),
}, open("model/metrics.json", "w"), indent=2)
print("best:", best, "| 5-fold CV accuracy: %.4f +/- %.4f" % (cv.mean(), cv.std()), "| saved to model/")
