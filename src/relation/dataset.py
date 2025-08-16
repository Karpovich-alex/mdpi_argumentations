import pandas as pd
import torch
from torch.utils.data import Dataset


class RelationDataset(Dataset):
    def __init__(self, relation_data, is_attacks, tokenizer, max_length):
        self.relation_data = relation_data
        # self.anns = torch.tensor(anns.values.astype("float32"))
        self.attack = torch.tensor(is_attacks.values).type(torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_csv(cls, path, tokenizer, max_length, sep=",", output_dim=1, **kwargs):
        df = pd.read_csv(path, sep=sep)
        relation_data = df[["arg_1", "arg_2"]]
        if output_dim == 1:
            is_attacks = df[["is_attacks"]]
        else:
            is_attacks = pd.get_dummies(df["is_attacks"])
        return cls(relation_data, is_attacks, tokenizer, max_length, **kwargs)

    def get_max_sentence_length(self):
        return self.relation_data.apply(lambda s: len(s)).max()

    def __len__(self):
        return len(self.relation_data)

    def __getitem__(self, idx):
        relation = self.relation_data.loc[idx].values
        attack = self.attack[idx]
        input_ids = []
        attention_masks = []
        for argument in relation:
            encoding = self.tokenizer(argument, truncation=True, padding='max_length', max_length=self.max_length,
                                      return_tensors='pt')
            input_ids.append(encoding['input_ids'].squeeze())
            attention_masks.append(encoding['attention_mask'].squeeze())
        # [input_ids_1, input_ids_2], [mask_1, mask_2], is_attack(0/1)
        return input_ids, attention_masks, attack

    def cut(self, start=0, stop=-1, step=1):
        self.relation_data = self.relation_data[start:stop:step]
        self.attack = self.attack[start:stop:step]


class RelationDatasetBert(RelationDataset):
    def __getitem__(self, idx):
        relation = self.relation_data.loc[idx].values
        attack = self.attack[idx]
        encoding = self.tokenizer(*relation, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')
        # [input_ids_1 + input_ids_2], [mask_1 + mask_2], is_attack(0/1)
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), attack


class RawDataset(RelationDataset):
    def __getitem__(self, idx):
        relation = self.relation_data.iloc[idx].T.values
        attack = self.attack[idx]
        # [sent_1, sent_2], is_attack(0/1)
        return relation, attack
