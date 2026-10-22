# Training data folders

## Image
Put CIFAKE (or another licensed real-vs-AI dataset) here:
data/image/train/REAL
data/image/train/FAKE
data/image/test/REAL
data/image/test/FAKE

Recommended first run: 3,000-5,000 images per class for train and 500-1,000 per class for test.
Do NOT mix test images into training.

## AI text
Create data/ai_text.csv with:
text,label
"human written example...",0
"AI generated example...",1

## News vs Not-News
Create data/input_type.csv with:
text,label
"Reuters article headline and body...",1
"Please grant me leave tomorrow...",0

1 = NEWS, 0 = NOT_NEWS
