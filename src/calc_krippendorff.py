import pandas as pd
import krippendorff


def get_annotation_by_words(path_to_ann):
    df = pd.read_csv(path_to_ann, sep="\t", encoding="UTF-8")
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


if __name__ == "__main__":
    path_to_ann_1 = r".\data\admsci5030125_boyarkin.tsv"
    words_1, ann_arr_1 = get_annotation_by_words(path_to_ann_1)

    path_to_ann_2 = r".\data\admsci5030125_makarova.tsv"
    words_2, ann_arr_2 = get_annotation_by_words(path_to_ann_2)

    assert len(words_1) == len(ann_arr_1) == len(words_2) == len(ann_arr_2), "lens are not equal"

    print(krippendorff.alpha(reliability_data=[ann_arr_1, ann_arr_2]))
