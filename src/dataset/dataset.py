import itertools

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils import read_from_json


def to_word_annotation(text, anns):
    text_word = []
    ann_word = []
    for sent, ann in zip(text, anns):
        sent_word = sent.split()
        text_word.extend(sent_word)
        ann_word.extend([ann for _ in range(len(sent_word))])
    return text_word, ann_word


class TextClassificationDataset(Dataset):
    def __init__(self, texts, anns, tokenizer, max_length, ds_type="sent"):
        self.texts = texts
        self.anns = anns
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ds_type = ds_type

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        ann = self.anns[idx]
        sent_ids = []
        sent_att_masks = []
        for sent in text:
            encoding = self.tokenizer(" ".join(sent), truncation=True, padding='max_length', max_length=self.max_length,
                                      return_tensors='pt')
            sent_ids.append(encoding['input_ids'].squeeze())
            sent_att_masks.append(encoding['attention_mask'].squeeze())

        return sent_ids, sent_att_masks, ann


class SentenceDataset(Dataset):
    def __init__(self, texts, anns, tokenizer, max_length):
        self.texts = texts
        self.anns = torch.tensor(anns.values).type(torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_json(cls, path, tokenizer, max_length, **kwargs):
        dataset = read_from_json(path)
        df = pd.DataFrame(dataset)
        df["mask_anns"] = df["anns"].apply(lambda a: list(map(lambda s: 0 if s == "0" else 1, a)))
        text = pd.Series(itertools.chain(*df["texts"]))
        annotation = pd.get_dummies(pd.Series(itertools.chain(*df["mask_anns"])))
        return cls(text, annotation, tokenizer, max_length, **kwargs)

    @classmethod
    def from_csv(cls, path, tokenizer, max_length, **kwargs):
        df = pd.read_csv(path)
        text = df["text"]
        annotation = pd.get_dummies(df["mask_ann"])
        return cls(text, annotation, tokenizer, max_length, **kwargs)

    def get_max_sentence_length(self):
        return self.texts.apply(lambda s: len(s)).max()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        ann = self.anns[idx].squeeze()

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask, ann
