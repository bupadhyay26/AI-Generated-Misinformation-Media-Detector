"""Download/prepare text datasets from Hugging Face.

Install once:
pip install datasets

This creates:
data/ai_text.csv
data/input_type.csv

If a dataset schema changes, inspect the printed columns and adjust the two
column names in this script.
"""
from pathlib import Path
import pandas as pd
from datasets import load_dataset

Path("data").mkdir(exist_ok=True)

# AI text: MDAIGT (20k; human/machine). We use a balanced subset.
ai=load_dataset("CogniSAL/MDAIGT")
parts=[]
for split in ai:
    d=ai[split].to_pandas()
    print("MDAIGT",split,d.columns.tolist(),len(d))
    parts.append(d)
df=pd.concat(parts,ignore_index=True)
text_col=next((c for c in ["text","content","sentence"] if c in df.columns),None)
label_col=next((c for c in ["label","class"] if c in df.columns),None)
if text_col is None or label_col is None:
    raise RuntimeError("MDAIGT columns not recognized; print columns above and update script.")
out=df[[text_col,label_col]].rename(columns={text_col:"text",label_col:"label"}).dropna()
# normalize labels
def norm(x):
    s=str(x).lower()
    if s in {"human","0","real"}: return 0
    if s in {"machine","ai","generated","1","fake"}: return 1
    try: return int(x)
    except: return None
out["label"]=out["label"].map(norm)
out=out.dropna(subset=["label"])
out["label"]=out["label"].astype(int)
a=out[out.label==0].sample(min(10000,(out.label==0).sum()),random_state=42)
b=out[out.label==1].sample(min(10000,(out.label==1).sum()),random_state=42)
pd.concat([a,b]).sample(frac=1,random_state=42).to_csv("data/ai_text.csv",index=False)
print("Wrote data/ai_text.csv",len(a),len(b))

# NEWS: AG News. Human/non-news: DailyDialog.
news=load_dataset("fancyzhx/ag_news",split="train").to_pandas()
news_text=(news["title"].fillna("")+" "+news["description"].fillna("")).str.strip() if "description" in news else news["text"].fillna("")
news_df=pd.DataFrame({"text":news_text,"label":1}).drop_duplicates("text")

dd=load_dataset("anezatra/dailydialog",split="train").to_pandas()
col="dialog" if "dialog" in dd.columns else "text"
non_df=pd.DataFrame({"text":dd[col].astype(str),"label":0}).drop_duplicates("text")
n=min(10000,len(news_df),len(non_df))
out2=pd.concat([news_df.sample(n,random_state=42),non_df.sample(n,random_state=42)]).sample(frac=1,random_state=42)
out2.to_csv("data/input_type.csv",index=False)
print("Wrote data/input_type.csv",len(out2))
