import csv
import json

import pandas as pd


def read_from_json(filename):
    with open(filename, 'r', encoding="UTF-8") as f:
        data = json.load(f)
    return data


def get_annotation_arr(path_to_ann, encoding="UTF-8"):
    texts = []
    anns = []
    with open(path_to_ann, "r", encoding=encoding) as file:
        for line in file.readlines():
            if line == "":
                continue
            splitted = line.strip().split("\t")
            if len(splitted) != 2:
                continue
            ann, text = splitted
            texts.append(text)
            anns.append(ann)
    return texts, anns


def get_annotation_arr_tsv(path_to_ann, encoding="UTF-8"):
    df = pd.read_csv(path_to_ann, sep="\t", encoding=encoding, quoting=csv.QUOTE_NONE)[["ann", "text"]]
    df_list = df.to_dict("list")
    return df_list["text"], df_list["ann"]
