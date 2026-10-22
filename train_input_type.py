"""Train NEWS vs NOT_NEWS classifier.
CSV: data/input_type.csv with text,label; 1=NEWS, 0=NOT_NEWS."""
import argparse,json
from pathlib import Path
import joblib,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
p=argparse.ArgumentParser(); p.add_argument("--data",default="data/input_type.csv"); args=p.parse_args()
df=pd.read_csv(args.data).dropna(subset=["text","label"]); df["label"]=df.label.astype(int)
tr,te=train_test_split(df,test_size=.2,stratify=df.label,random_state=42)
pipe=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True)),
               ("model",LogisticRegression(max_iter=2000,class_weight="balanced"))])
pipe.fit(tr.text,tr.label); pred=pipe.predict(te.text)
pr,re,f1,_=precision_recall_fscore_support(te.label,pred,average="binary",zero_division=0)
Path("model").mkdir(exist_ok=True); joblib.dump(pipe,"model/input_type_model.joblib",compress=3)
m={"task":"NEWS vs NOT_NEWS","train":len(tr),"test":len(te),"accuracy":float(accuracy_score(te.label,pred)),
"precision":float(pr),"recall":float(re),"f1":float(f1),"confusion_matrix":confusion_matrix(te.label,pred).tolist()}
Path("model/input_type_metrics.json").write_text(json.dumps(m,indent=2)); print(json.dumps(m,indent=2))
