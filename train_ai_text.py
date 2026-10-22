"""Train AI-vs-human text detector.
Input CSV: data/ai_text.csv with columns: text,label
label: 0=human-written, 1=AI-generated

Example:
.venv\Scripts\python train_ai_text.py --data data/ai_text.csv --max-per-class 10000
"""
import argparse, json
from pathlib import Path
import joblib, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

p=argparse.ArgumentParser()
p.add_argument("--data", default="data/ai_text.csv")
p.add_argument("--max-per-class", type=int, default=10000)
args=p.parse_args()

df=pd.read_csv(args.data)
required={"text","label"}
if not required.issubset(df.columns): raise SystemExit("CSV must contain text,label")
df=df.dropna(subset=["text","label"]).copy()
df["label"]=df["label"].astype(int)
df=df[df["text"].str.split().str.len()>=25]
parts=[]
for lab in [0,1]:
    x=df[df.label==lab]
    if len(x)>args.max_per_class: x=x.sample(args.max_per_class,random_state=42)
    parts.append(x)
df=pd.concat(parts).sample(frac=1,random_state=42).reset_index(drop=True)

tr,te=train_test_split(df,test_size=.2,stratify=df.label,random_state=42)
word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=80000,sublinear_tf=True)
char=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True)
Xtr=hstack([word.fit_transform(tr.text),char.fit_transform(tr.text)])
Xte=hstack([word.transform(te.text),char.transform(te.text)])
clf=LogisticRegression(max_iter=2500,class_weight="balanced",C=2.0)
clf.fit(Xtr,tr.label)
pred=clf.predict(Xte)
acc=accuracy_score(te.label,pred)
pr,re,f1,_=precision_recall_fscore_support(te.label,pred,average="binary",zero_division=0)
cm=confusion_matrix(te.label,pred).tolist()
Path("model").mkdir(exist_ok=True)
joblib.dump({"word_vectorizer":word,"char_vectorizer":char,"model":clf}, "model/ai_text_model.joblib", compress=3)
metrics={"task":"AI-generated vs human-written text","samples":len(df),"train":len(tr),"test":len(te),
         "accuracy":float(acc),"precision":float(pr),"recall":float(re),"f1":float(f1),"confusion_matrix":cm,
         "labels":{"0":"HUMAN-WRITTEN","1":"AI-GENERATED"}}
Path("model/ai_text_metrics.json").write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
