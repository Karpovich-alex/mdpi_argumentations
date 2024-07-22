import csv
import os

import numpy as np
import pandas as pd
import krippendorff


def get_annotation_by_words(path_to_ann):
    df = pd.read_csv(path_to_ann, sep="\t", encoding="UTF-8", quoting=3)
    df["ann"] = df["ann"].apply(lambda x: x > 0).astype(int)
    texts, anns = df["text"].tolist(), df["ann"].tolist()
    words = []
    ann_arr = []
    for sent, ann in zip(texts, anns):
        words_in_sentence = sent.split()
        for word in words_in_sentence:
            words.append(word)
            ann_arr.append(ann)
    assert len(words) == len(ann_arr), "lens are not equal"
    return words, ann_arr


def find_difference(str1, str2):
    for idx, el in enumerate(zip(str1, str2)):
        el1, el2 = el
        if el1 != el2:
            return "#" + str(idx) + ":  " + el1 + "  ---  " + el2


def get_pairs(files) -> list[tuple[str, str]]:
    pairs = []
    for idx, file1 in enumerate(files):
        file1_article = file1.split("_")[0]
        for idx2 in range(idx + 1, len(files)):
            file2 = files[idx2]
            file2_article = file2.split("_")[0]
            if file1_article == file2_article:
                pairs.append((file1, file2))
            break
    return pairs

def calc_krippendorff(path_to_ann_1, path_to_ann_2):
    words_1, ann_arr_1 = get_annotation_by_words(path_to_ann_1)
    words_2, ann_arr_2 = get_annotation_by_words(path_to_ann_2)

    assert len(words_1) == len(ann_arr_1) == len(words_2) == len(
        ann_arr_2), f"{os.path.split(path_to_ann_1)[-1]} lens are not equal: {len(words_1)} != {len(ann_arr_1)} != {len(words_2)}. first difference: {find_difference(words_1, words_2)}"

    return krippendorff.alpha(reliability_data=[ann_arr_1, ann_arr_2])


if __name__ == "__main__":
    # path_to_ann_1 = r".\data\w14081258_makarova.tsv"
    # path_to_ann_2 = r".\data\w14081258_perova.tsv"
    # print(calc_krippendorff_for_one(path_to_ann_1, path_to_ann_2))

    path_to_reviews = r".\data"
    review_files = list(filter(lambda x: x.endswith(".tsv"), os.listdir(path_to_reviews)))
    review_pairs = get_pairs(review_files)
    krippendorff_array = np.array([])
    for pair in review_pairs:
        krippendorff_array = np.append(krippendorff_array, calc_krippendorff(os.path.join(path_to_reviews, pair[0]), os.path.join(path_to_reviews, pair[1])))
    print("mean:", np.mean(krippendorff_array))
    print("std:", np.std(krippendorff_array))

