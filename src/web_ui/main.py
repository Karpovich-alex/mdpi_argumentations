import itertools
import json
import logging
import operator
import os
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from functools import reduce

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from annotated_text import annotated_text
from streamlit_agraph import agraph, Node, Edge, Config
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import TensorDataset
from utils import Convertor

from src.web_ui.ui_utils import get_colors_from_palette, get_closed_arguments

ARGUMENT_TYPES = {"": "text", "A": "Argument"}

RELATION_TRESHOLD = 0.5

MODELS_SELECTION_ARGUMENT = [{"name": "LSTM",
                              "path": r"<path-to-checkpoint>\LSTMClassifier\epoch=99-step=41500.ckpt"}]
MODELS_SELECTION_RELATION = [{"name": "LSTM",
                              "path": r"<path-to-checkpoint>\LSTMClassifier\epoch=36-step=2664.ckpt"}]


@st.cache_resource(max_entries=1, show_spinner="Loading relation model")
def get_relation_model(path_to_checkpoint):
    from src.relation.lstm_model import LSTMClassifier
    model = LSTMClassifier.load_from_checkpoint(
        path_to_checkpoint, )  # .to("cuda")
    model.eval()
    return model


@st.cache_resource(max_entries=1, show_spinner="Loading argument model")
def get_argument_model(path_to_checkpoint):
    from src.models.lstm_model import LSTMClassifier
    model = LSTMClassifier.load_from_checkpoint(
        path_to_checkpoint, )  # .to("cuda")
    model.eval()
    return model


@st.cache_resource(max_entries=1, show_spinner="Loading argument model")
def get_tokenizer(tokenizer_name):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


@st.cache_data(max_entries=5)
def prepare_text(text, batch_size=64, device="cuda") -> tuple[list[str], DataLoader, TensorDataset]:
    from utils import Convertor

    max_length = 110
    tokenizer_name = "distilbert-base-uncased"
    tokenizer = get_tokenizer(tokenizer_name)
    converter = Convertor()
    sentences = converter.apply_pipeline(text)
    encoding = tokenizer(sentences, truncation=True, padding='max_length', max_length=max_length,
                         return_tensors='pt')
    input_ids = encoding['input_ids'].squeeze().to(device)
    attention_mask = encoding['attention_mask'].squeeze().to(device)
    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return sentences, loader, dataset


@st.cache_data(show_spinner="Extracting arguments")
def extract_arguments(_model, text, model_name):
    sentences, loader, dataset = prepare_text(text, device=_model.device)
    outputs = []

    with torch.no_grad():
        for batch in loader:
            output = torch.argmax(_model(*batch).to("cpu"), dim=1).tolist()
            outputs.extend(output)
    return sentences, dataset, outputs


# @st.cache_data(max_entries=3, show_spinner="Processing data")
def get_relation_dataset(review_arguments: list[tuple[TensorDataset, pd.DataFrame]],
                         reviewers_metadata) -> tuple[
    ConcatDataset, list[tuple[list[int], list[int]]], np.array]:
    review_idx_comb = []  # [(ids_text_round_N, ids_text_round_N-1), (ids_text_round_N-1, ids_text_round_N-2), ...]
    review_datasets = []
    skip_mask = []
    for idx in range(len(review_arguments) - 1, 0, -1):
        metadata_1, metadata_2 = reviewers_metadata[idx], reviewers_metadata[idx - 1]
        # Пропускаем пары ревью, которые написаны разными ревьюерами, так как между ними не может быть связи
        # Добавляем в skip_mask 1 для пропуска этой комбинации ревью при разборе результата модели
        if metadata_1["reviewer_num"] != metadata_2["reviewer_num"] or metadata_1["round_side_str"] == metadata_2[
            "round_side_str"]:
            skip_mask.append(1)
            continue
        skip_mask.append(0)
        dataset_1, df_1 = review_arguments[idx]
        dataset_2, df_2 = review_arguments[idx - 1]

        df_1 = df_1[df_1["class"] != 0][["old_index", "sentence"]]
        df_2 = df_2[df_2["class"] != 0][["old_index", "sentence"]]

        idx_comb = list(itertools.product(df_1["old_index"].tolist(), df_2["old_index"].tolist()))

        idx_comb1 = []
        idx_comb2 = []
        for pair_idx in idx_comb:
            idx1, idx2 = pair_idx
            idx_comb1.append(idx1)
            idx_comb2.append(idx2)
        review_idx_comb.append((idx_comb1, idx_comb2))

        ds1 = dataset_1[idx_comb1]
        ds2 = dataset_2[idx_comb2]

        dataset = TensorDataset(ds1[0], ds2[0], ds1[1], ds2[1])
        review_datasets.append(dataset)
    dataset = ConcatDataset(review_datasets)

    return dataset, review_idx_comb, np.array(skip_mask, dtype=bool)


@st.cache_data(max_entries=3, show_spinner="Extracting relations")
def extract_relations(_review_arguments: list[tuple[TensorDataset, pd.DataFrame]],
                      reviewers_metadata,
                      _model, model_name, text_sample_1, text_sample_2, batch_size=64) -> tuple[
    list[tuple[list[int], list[int]]], np.array, np.array]:
    dataset, review_idx_comb, skip_mask = get_relation_dataset(_review_arguments, reviewers_metadata)

    def collate_fn(batch):
        input_ids = []
        attention_mask = []
        for b in batch:
            input_ids.append(torch.stack((b[0], b[1])))
            attention_mask.append(torch.stack((b[2], b[3])))
        input_ids = torch.stack(input_ids).permute(1, 0, 2)
        attention_mask = torch.stack(attention_mask).permute(1, 0, 2)
        return input_ids, attention_mask

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)  # collate_fn=collate_fn

    outputs = []

    with torch.no_grad():
        for batch in loader:
            output = _model(*batch).to("cpu")
            # TODO: remove threshold
            # output = torch.where(output > 0.5, 1, 0)
            outputs.extend(output.reshape(-1).tolist())
    return review_idx_comb, np.array(outputs), skip_mask


def process_text_output(review_arguments: list[tuple[..., pd.DataFrame]], mapping):
    idx_to_annotation_str = list(mapping.keys())
    text_annotation = []
    for review_data in review_arguments:
        _, ann_df = review_data
        ann_text = []
        for x in ann_df[["class", "sentence"]].to_dict('tight')["data"]:
            ann, text = x
            ann_str = idx_to_annotation_str[ann]
            if ann_str:
                ann_text.append((text + " ", ann_str))
            else:
                ann_text.append(text + " ")
            # ann_text.append(" ")
        text_annotation.append(ann_text)
    # ann_text.pop()
    return text_annotation


@st.cache_data(max_entries=3, show_spinner="Processing data")
def get_relation_df(_review_arguments, review_idx_comb, skip_mask, is_relations) -> tuple[pd.DataFrame, pd.DataFrame]:
    relation_data = {"Argument 1 ID": [],
                     "Argument 1": [],
                     "Argument 2 ID": [],
                     "Argument 2": [],
                     "Probability": None}
    sentences = pd.DataFrame()
    probability_skip_mask = []
    for idx_comb, review_idx, need_to_skip in zip(review_idx_comb, range(len(_review_arguments) - 1, 0, -1), skip_mask):
        # Пропускаем невалидные пары, тк они принадлежат разным рецензентам
        if need_to_skip:
            print(len(idx_comb[0]))
            probability_skip_mask.extend([False for _ in range(len(idx_comb[0]))])
            continue
        probability_skip_mask.extend([True for _ in range(len(idx_comb[0]))])
        # Выбираем соответствующие DF для 1 и 2 аргумента
        df_1 = _review_arguments[review_idx][1]
        df_2 = _review_arguments[review_idx - 1][1]
        # Выбираем ID аргументов
        arg_ids_1, arg_ids_2 = idx_comb
        arg_ids_1, arg_ids_2 = torch.tensor(arg_ids_1), torch.tensor(arg_ids_2)

        df_1 = df_1.set_index("old_index").loc[arg_ids_1]
        df_2 = df_2.set_index("old_index").loc[arg_ids_2]

        relation_data["Argument 1 ID"].extend(df_1["ID"].tolist())
        relation_data["Argument 1"].extend(df_1["sentence"].tolist())
        relation_data["Argument 2 ID"].extend(df_2["ID"].tolist())
        relation_data["Argument 2"].extend(df_2["sentence"].tolist())

        sentences = pd.concat([sentences, df_1[["sentence", "text_id", "ID"]].drop_duplicates()])
        sentences = pd.concat([sentences, df_2[["sentence", "text_id", "ID"]].drop_duplicates()])
    relation_data["Probability"] = is_relations[probability_skip_mask]
    return pd.DataFrame(relation_data), sentences


@st.cache_data(max_entries=3, show_spinner="Processing data")
def filter_relation_dataset(df, threshold):
    return df[df["Probability"] > threshold]


@st.cache_data(max_entries=3, show_spinner="Calculating probabilities chart")
def get_relation_probabilities(relation_df, min_value, bins=150):
    count, division = np.histogram(relation_df["Probability"], bins=bins,
                                   range=(min_value, 1.))
    line_df = pd.DataFrame({"treshold": division[:-1][::-1], "count": np.cumsum(count[::][::-1])})
    return line_df


@st.cache_data(max_entries=3, show_spinner="Generating json")
def generate_arguments_pairs(df, sent_df):
    data = {"argument_sets": defaultdict(dict),
            "attack_pairs": []}
    # paper_sent_id = "paper"
    paper_sent_id = "0.0.0.0a"
    data["argument_sets"]["Author"][paper_sent_id] = "Paper"
    for sent_data in sent_df.to_dict("tight")["data"]:
        sent, text_id, sent_id = sent_data
        reviewer_num, round_num, side_alias = text_id.split(".")
        # if its first round and reviewer side then append arguments attacks the paper
        if side_alias == "R" and round_num == "1":
            data["attack_pairs"].append([sent_id, paper_sent_id])

        if side_alias == "A":
            side = "Author"
        else:
            side = f"Reviewer_{reviewer_num}"

        data["argument_sets"][side][sent_id] = sent

    pairs = df[["Argument 1 ID", "Argument 2 ID"]].to_dict("split")["data"]
    data["attack_pairs"].extend(pairs)
    return data


@st.cache_data
def get_result_name():
    return f"relations_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def sort_key_arguments(x):
    if isinstance(x, tuple):
        x = x[0]
    # reviewer_num, round_num, review_side = x.split(".")
    # review_side = '2' if review_side == "A" else '1'
    # return reviewer_num + round_num + review_side
    return 0 if x.startswith("A") else int(x.split("_")[1])


@st.cache_data
def get_graph(data, closed_args, max_label_length=15):
    arguments = data['argument_sets']
    pairs = data['attack_pairs']
    nodes = []
    edges = []
    color_map = get_colors_from_palette(len(arguments))

    for argument_idx, (argument_id, arguments_data) in enumerate(sorted(arguments.items(), key=sort_key_arguments)):
        st.markdown(f'<div>'
                    f'<p style="display:inline-block;padding-right:7px;">Side: {argument_id.replace("_", " ")}</p>'
                    f'<div style="display:inline-block;background-color:{color_map[argument_idx]};width:20px;height:20px;"></div>'
                    f'</div>', unsafe_allow_html=True)

        for text_id, argument_text in arguments_data.items():
            nodes.append(Node(id=text_id,
                              label=argument_text[:max_label_length] + "...",
                              title='#' + text_id + ' ' + argument_text,
                              size=10,
                              color=color_map[argument_idx],
                              shape="dot" if text_id != "paper" and text_id.split(".")[2] == "A" else "triangle",
                              group=text_id.split(".")[0] if text_id != "paper" else "A"))

    closed_arg1_ids = set(closed_args)
    for arg in pairs:
        edges.append(Edge(arg[0], arg[1]))

    for node in nodes:
        if node.id not in closed_arg1_ids:
            node.shape = "dot"
        if node.id == "paper" or node.id == "0.0.0a" or node.id == "0.0.0.0a":
            node.shape = "square"
    return nodes, edges


@st.cache_data
def get_review_result(data, threshold) -> list[str]:
    print("Get review result:", data)
    closed_args = get_closed_arguments(data)
    return closed_args


def st_normal():
    _, col, _ = st.columns([1, 2, 1])
    return col


HORIZONTAL_STYLE = """
<style class="hide-element">
    /* Hides the style container and removes the extra spacing */
    .element-container:has(.hide-element) {
        display: none;
    }
    /*
        The selector for >.element-container is necessary to avoid selecting the whole
        body of the streamlit app, which is also a stVerticalBlock.
    */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) {
        display: flex;
        flex-direction: row !important;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: baseline;
    }
    /* Buttons and their parent container all have a width of 704px, which we need to override */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) div {
        width: max-content !important;
    }
    /* Just an example of how you would style buttons, if desired */
    /*
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) button {
        border-color: red;
    }
    */
</style>
"""


@contextmanager
def st_horizontal():
    st.markdown(HORIZONTAL_STYLE, unsafe_allow_html=True)
    with st.container():
        st.markdown('<span class="hide-element horizontal-marker"></span>', unsafe_allow_html=True)
        yield


def main():
    def plus_round_func():
        st.session_state.reviewer_number += 1

    def minus_round_func():
        st.session_state.reviewer_number -= 1

    def runbtn_state_func():
        st.session_state.runbtn_state = True

    st_normal().__enter__()
    st.title('Arguments extraction')

    st.subheader('Models data')

    arg_model_data = st.selectbox("Argument extraction model", options=MODELS_SELECTION_ARGUMENT,
                                  format_func=lambda x: x["name"])
    rel_model_data = st.selectbox("Relation extraction model", options=MODELS_SELECTION_RELATION,
                                  format_func=lambda x: x["name"])

    arg_model_name = arg_model_data["name"]
    rel_model_name = rel_model_data["name"]

    arg_model_path = arg_model_data["path"]
    rel_model_path = rel_model_data["path"]

    st.subheader('Input files')
    st.text("""
    It is required to add files for each reviewer separately, if the discussion has several rounds, it is required to add them in a row: reviewer's file for round 1, author's response for round 1, reviewer's file for round 2, author's response for round 2, etc.
    You can check the correspondence of files and their type in the tab 'See review scheme'
    """)
    if 'reviewer_number' not in st.session_state:
        st.session_state.reviewer_number = 1
    st.text("To add more reviewers press +")
    with st_horizontal():
        st.button(":heavy_plus_sign:", on_click=plus_round_func)
        st.button(":heavy_minus_sign:", on_click=minus_round_func, disabled=(st.session_state.reviewer_number <= 1))

    reviewer_files = []
    for i in range(st.session_state.reviewer_number):
        reviewer_files.append(st.file_uploader(f"Reviewer {i + 1} files:", type=["txt", "doc", "pdf", "docx"],
                                               accept_multiple_files=True))
    is_files_loaded = reduce(operator.and_, map(bool, reviewer_files), True)

    if "runbtn_state" not in st.session_state or not is_files_loaded:
        st.session_state.runbtn_state = False

    with st.expander("See review scheme"):
        for idx, round_files in enumerate(reviewer_files):
            round_text = f"**Reviewer {idx + 1}**\n"
            for file_idx, file in enumerate(round_files):
                if file_idx % 2 == 0:
                    round_text += f"- __Round {file_idx // 2 + 1}__\n"
                if file_idx % 2:
                    round_text += f"\t- **Author file:** {file.name}\n"
                else:
                    round_text += f"\t- **Reviewer file:** {file.name}\n"
            if not round_files:
                round_text += "- No review selected"
            st.markdown(round_text.rstrip())

    # st.write(is_files_loaded)
    # st.write(st.session_state.runbtn_state)
    st.button("Run", type="primary", disabled=not is_files_loaded, help="Choose files to run",
              on_click=runbtn_state_func)

    if is_files_loaded and st.session_state.runbtn_state:

        # Load models
        argument_model = get_argument_model(arg_model_path)
        relation_model = get_relation_model(rel_model_path)

        convertor = Convertor()
        review_arguments = []
        no_arg_found = False
        article_relations = pd.DataFrame()
        article_sentences = pd.DataFrame()
        reviewers = []
        reviewers_metadata = []
        review_data = defaultdict(lambda: defaultdict(list))
        annotation_idx = 0
        for idx, round_files in enumerate(reviewer_files):
            reviewer_num = idx + 1
            reviewers.append([])
            round_number = 1
            for file_idx, file in enumerate(round_files):
                if file_idx % 2 == 0:
                    round_number = file_idx // 2 + 1
                round_side = "A" if file_idx % 2 else "R"
                reviewers[idx].append(round_number)
                reviewers_metadata.append(
                    {"round": round_number, "round_side": round_side,
                     "round_side_str": "Author" if round_side == "A" else "Reviewer",
                     "reviewer_num": reviewer_num,
                     "annotation_idx": annotation_idx})
                review_data[reviewer_num][round_number].append(reviewers_metadata[-1])
                annotation_idx += 1

                file_ext = os.path.splitext(file.name)[-1]
                # Read  files
                # if idx == 0 and file_idx == 0:
                #     print(file)
                #     print(file.read().encode("UTF-8"))
                text = convertor.extension[file_ext](file)
                sentences, dataset, sent_class = extract_arguments(argument_model, text, arg_model_name)
                if sum(sent_class) == 0:
                    logging.warning("no arguments found in article round %s for %s type reviewer %s", round_number,
                                    round_side, reviewer_num)
                    if idx + 1 != len(round_files):
                        no_arg_found = True
                        # break
                    # else:
                    #     continue
                ann_df = pd.DataFrame({"sentence": sentences, "class": sent_class})
                ann_df = ann_df.reset_index(drop=False).rename({"index": "old_index"}, axis=1)
                # ann_df = ann_df[ann_df["class"] != 0][["sentence"]].reset_index(drop=False).rename(
                #     {"index": "old_index"}, axis=1)
                # text_id = Reviewer_number.Round_number.Round_side.Sentence_number
                ann_df["text_id"] = str(reviewer_num) + "." + str(round_number) + "." + str(round_side)
                ann_df["ID"] = ann_df["text_id"] + "." + ann_df.reset_index()["index"].astype(str)
                review_arguments.append((dataset, ann_df))
            if no_arg_found:
                logging.warning("skipping reviewer_num %s no arguments in round %s", reviewer_num, round_number)
                continue
            if len(review_arguments) == 1:
                logging.warning("skipping reviewer_num %s one round argument found", reviewer_num)
                continue
            review_idx_comb, is_relations, skip_mask = extract_relations(review_arguments, reviewers_metadata,
                                                                         relation_model, rel_model_name,
                                                                         text_sample_1=file.name + str(
                                                                             reviewer_num) + str(
                                                                             round_number) + "1",
                                                                         text_sample_2=file.name + str(
                                                                             reviewer_num) + str(
                                                                             round_number) + "2")
            review_relations, sentences_df = get_relation_df(review_arguments, review_idx_comb, skip_mask, is_relations)
            article_relations = pd.concat([article_relations, review_relations])
            article_sentences = pd.concat([article_sentences, sentences_df])

        st_normal().__exit__(None, None, None)
        st.subheader('Review analysis')
        tab1, tab2, tab3, tab4 = st.tabs(["Text", "Arguments", "Relations", "Graph"])
        # print(review_arguments)
        text_annotations = process_text_output(review_arguments, ARGUMENT_TYPES)

        print_legend = lambda: annotated_text(*((v, k) for k, v in ARGUMENT_TYPES.items() if k != ""))
        # print(len(text_annotations))
        with tab1:
            print_legend()
            text_tabs = st.tabs([f"Reviewer {x}" for x in review_data.keys()])
            for tab, metadata in zip(text_tabs, review_data.values()):
                with tab:
                    current_tabs = st.tabs([f"Round {x}" for x in metadata])
                    for cur_tab, round_data in zip(current_tabs, metadata.values()):
                        with cur_tab:
                            for cur_round in round_data:
                                st.markdown(f"#### {cur_round['round_side_str']}")
                                annotated_text(*text_annotations[cur_round["annotation_idx"]])
        with tab2:
            print_legend()
            st.write("On this tab you can see all the extracted arguments for all the review files")
            text_tabs = st.tabs([f"Reviewer {x}" for x in review_data.keys()])
            for tab, metadata in zip(text_tabs, review_data.values()):
                with tab:
                    current_tabs = st.tabs([f"Round {x}" for x in metadata])
                    for cur_tab, rounds_data in zip(current_tabs, metadata.values()):
                        with cur_tab:
                            for round_data in rounds_data:
                                _, ann_df = review_arguments[round_data["annotation_idx"]]
                                with st_normal():
                                    st.markdown(f"#### {round_data['round_side_str']}")
                                    st.dataframe(ann_df[ann_df["class"] == 1].set_index("ID")
                                                 .drop(["class", "text_id", "old_index"], axis=1),
                                                 height=750 if ann_df.shape[0] >= 20 else None, width=2500)

        with tab3:
            st.write(
                "On this tab you can see the extracted relations depending on the threshold set for the presence of a relation between the arguments."
                "You can download the extracted relations and select an argument to see all the relationships with it")
            relation_threshold = st.number_input("Relation threshold", value=RELATION_TRESHOLD, min_value=0.,
                                                 max_value=1., step=0.01)
            filter_only_relations = st.toggle("Apply threshold", value=True,
                                              help=f"Apply threshold to probabilities to filter the relations")
            current_relation_threshold = relation_threshold if filter_only_relations else 0
            min_hist_value = relation_threshold if filter_only_relations else 0
            line_df = get_relation_probabilities(article_relations, min_hist_value)
            with st.expander("Probabilities chart", expanded=True, icon=":material/bar_chart:"):
                chart = alt.Chart(line_df).mark_bar().encode(
                    x=alt.X("treshold:Q", scale=alt.Scale(padding=0)),
                    y=alt.Y("count:Q")
                )
                st.altair_chart(chart, use_container_width=True)
            current_df = filter_relation_dataset(article_relations,
                                                 threshold=relation_threshold) if filter_only_relations else article_relations
            # current_df.to_csv("filter_relation_dataset.csv", index=False)
            st.write(f"Total rows: **{current_df.shape[0]}**")
            event = st.dataframe(current_df,
                                 height=750 if current_df.shape[0] >= 20 else None,
                                 width=2500,
                                 selection_mode="single-row",
                                 key="df",
                                 on_select="rerun"
                                 )
            # Download button
            arg_data = generate_arguments_pairs(current_df, article_sentences)
            json_data = json.dumps(arg_data)
            json_name = get_result_name()
            st.download_button(
                label="Download json",
                data=json_data,
                file_name=f"{json_name}.json",
                mime="application/json",
                help="The current state of the threshold is applied to the result file"
            )

            st.header("Selected relations")
            if event.selection.rows:
                row_id = event.selection.rows[0]
                selected_df = current_df[current_df["Argument 1"] == current_df.iloc[row_id, :]["Argument 1"]]
                st.write(f"Selected Argument 1 ID: **{selected_df.iloc[0]['Argument 1 ID']}**")
                st.write(f"Total rows: **{selected_df.shape[0]}**")

                st.dataframe(selected_df.drop('Argument 1 ID', axis=1), width=2500, )
            else:
                st.write("No rows selected  ")

        with (tab4):
            #  Reviewer_number.Round_number.Round_side.Sentence_number
            st.write(f"**Relation threshold:** {current_relation_threshold}")
            st.write(
                "**Arguments displays in format:** #<Reviewer number>.<Round number>.<Round side>.<Sentence number> Argument text")
            st.write(
                "Main Paper argument is displayed with a square. Closed arguments are displayed with a circle, unclosed arguments are displayed with a triangle")
            closed_args = get_review_result(arg_data, current_relation_threshold)

            main_arg_text = "**not accepted** (not all attacking arguments are closed)"
            if "paper" in closed_args or \
                    "0.0.0a" in closed_args or \
                    "0.0.0.0a" in closed_args:
                main_arg_text = "**accepted** (all attacking arguments are closed)"

            st.write(f"Main Paper argument is {main_arg_text}")
            col1, col2 = st.columns(2, border=True)
            with col2:
                nodes, edges = get_graph(arg_data, closed_args)
            # config = Config(width=1000,
            #                 directed=True,
            #                 nodeHighlightBehavior=False,
            #                 highlightColor="#F7A7A6",  # or "blue"
            #                 # collapsible=True,
            #                 node={'labelProperty': 'label'},
            #                 physics=False,
            #                 hierarchical=False,
            #                 # groups=True
            #                 )
            config = Config(width=1000,
                            minVelocity=0.5,
                            maxVelocity=2,
                            hierarchical=True,
                            nodeSpacing=1,
                            levelSeparation=5,
                            treeSpacing=10,
                            parentCentralization=False,
                            )
            with col1:
                agraph(nodes=nodes,
                       edges=edges,
                       config=config)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Arguments extractor", page_icon=":mag:", layout="wide"
    )
    main()
