import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_annotated_text_html(path_to_ann, path_to_html, legend=None):
    def colors_text(text, text_class):
        return (f"<class-{text_class}>{text}</class-{text_class}>")

    def gradient_colors(min_v, max_v):
        cmap = plt.get_cmap("Pastel1")
        norm = mcolors.Normalize(vmin=min_v, vmax=max_v)
        return [mcolors.to_hex(cmap(norm(number))) for number in range(min_v, max_v + 1)]

    df = pd.read_csv(path_to_ann, sep="\t", encoding="UTF-8")
    texts, anns = df["text"].tolist(), df["ann"].tolist()
    anns = np.array(anns)
    unique_anns = np.unique(anns)
    anns_colors = gradient_colors(1, len(unique_anns))

    file = open(path_to_html, "w", encoding="UTF-8")

    file.write("<html>")
    color_classes = [f"class-{idx + 1} {{background-color: {c};}}" for idx, c in enumerate(anns_colors)]
    color_classes.append("class-0 {background-color: #FFFFFF;}")
    file.write(f"<style> html {{ font-family: \"Times New Roman\";}} {' '.join(color_classes)} </style>")
    file.write("<div>Legend: ")

    if legend is None:
        legend = unique_anns
    for idx, leg in enumerate(legend):
        file.write(colors_text(leg, idx) + " ")
    file.write(f"\tTotal classes num: {len(unique_anns)}")
    file.write("</div><hr><div>")
    for ann, text in zip(anns, texts):
        file.write(colors_text(text + " ", ann))

    file.write("</div></html>")
    file.close()



if __name__ == "__main__":
    save_annotated_text_html(
        r"./data/life5021427_makarova.tsv",
        "./life5021427_makarova.html",
        legend=["Non-argument", "Author argument", "Reviewer argument"])
