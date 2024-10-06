import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import get_annotation_arr


def make_dataset(path="./data"):
    articles = []
    annotator = []
    texts = []
    anns = []
    iterator = tqdm(os.listdir(path))
    for article in iterator:
        if not article.endswith('.ann'):
            continue
        iterator.set_postfix_str(article)
        text, ann = get_annotation_arr(os.path.join(path, article))
        texts.append(text)
        anns.append(ann)
        article_name, annotator_name = article.split("_")
        annotator_name = annotator_name.split(".")[0]
        articles.append(article_name)
        annotator.append(annotator_name)
    return articles, annotator, texts, anns


# Read dataset files and combine them
articles, annotator, texts, anns = make_dataset(path="./data")

# Check: Length of texts is equals to length of annotations
for t, a in zip(texts, anns):
    assert len(t) == len(a)

# Number of annotations
print(len(articles), len(annotator), len(texts), len(anns))

# Number of sentences
print(sum(len(t) for t in texts))

# Convert dataset to pandas DataFrame
df = pd.DataFrame({"text": texts, "ann": anns, "article": articles, "annotator": annotator})
# Split sentences to rows
df = df.explode(["text", "ann"])
df["ann"] = df["ann"].astype(int)
# Create a column = sentence annotation or no
df["mask_ann"] = df["ann"].apply(lambda a: 0 if a == 0 else 1)

# Divide dataset into train, validation and test
test_size, val_size = 0.2, 0.1
random_state = 42

df_train, df_test = train_test_split(df[["text", "ann", "mask_ann"]], test_size=test_size, random_state=random_state,
                                     stratify=df["mask_ann"])
df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=random_state,
                                    stratify=df_train["mask_ann"])

# Save dataset
if not os.path.exists("./dataset/sentence"):
    os.mkdir("./dataset/sentence")

df_train.to_csv("./dataset/sentence/train.csv", index=False)
df_val.to_csv("./dataset/sentence/val.csv", index=False)
df_test.to_csv("./dataset/sentence/test.csv", index=False)
