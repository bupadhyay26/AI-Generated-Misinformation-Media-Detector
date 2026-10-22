# DATASET COLLECTION — DO THIS ONCE

## A. IMAGE DATASET (required)
Use CIFAKE. It has REAL and AI-generated classes, with 100,000 training images (50k/class) and 20,000 test images (10k/class). The original fake class was generated with Stable Diffusion v1.4.
Official project: https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images
Kaggle: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

For the first training run, you do NOT need all 120k:
- train/REAL: 3000-5000
- train/FAKE: 3000-5000
- test/REAL: 500-1000
- test/FAKE: 500-1000

Do not copy test images into train.

Folder:
data/image/train/REAL
data/image/train/FAKE
data/image/test/REAL
data/image/test/FAKE

## B. AI TEXT DATASET (required)
Use a labeled human-vs-AI dataset. MDAIGT is a current public benchmark with human and machine-generated text across news and academic domains:
https://huggingface.co/datasets/CogniSAL/MDAIGT

Create:
data/ai_text.csv
Columns exactly:
text,label
label 0 = HUMAN
label 1 = AI

Start with 5000-10000 examples per class. Keep a separate 20% test split through train_ai_text.py.

Alternative: AITDNA is useful for realistic human-AI interaction/edited-text evaluation:
https://huggingface.co/datasets/UKPLab/AITDNA

## C. NEWS vs NOT-NEWS DATASET (required)
Use:
- AG News for NEWS examples: https://huggingface.co/datasets/fancyzhx/ag_news
- DailyDialog for human conversational/non-news examples: https://huggingface.co/datasets/anezatra/dailydialog

Build data/input_type.csv:
text,label
label 1 = NEWS
label 0 = NOT_NEWS

Recommended first run:
- 5000-10000 NEWS
- 5000-10000 NOT_NEWS

Important: Do not use only personal messages as NOT_NEWS. Include questions, conversations, requests, greetings, instructions and ordinary statements.

## D. VIDEO
Do NOT collect a giant video dataset for the first version.
The trained image detector is applied to sampled frames and aggregated.
For a later dedicated-video extension, FaceForensics++ is a research dataset, but access requires accepting its terms and the dataset is very large:
https://github.com/ondyari/FaceForensics

## E. DATASET REPORTING
In the report record:
- dataset name
- source URL
- license/terms
- number of samples used
- class distribution
- train/test split
- preprocessing
- model architecture
- accuracy, precision, recall, F1
- limitations

Never claim that the model proves provenance. Report "AI-generated" as a classifier prediction.
