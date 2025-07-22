# mdpi_argumentations

## Description

- Total number of reviews: 115
- Each review has been annotated two times by different annotators.
- Total number of annotations: 164

## Dataset Structure

{article_review_name}.txt - file with collected text from all files from the review  
{article_review_name}_{reviewer}.tsv - file with annotation of the review

### Annotation structure (*.tsv)

Stores all sentences from the review. If sentence is not a review, then all columns will be None except _ann_ and _text_

| Column name | Description                                                |
|-------------|------------------------------------------------------------|
| side        | side of the argument.                                      |
| opponent    | opponent of the argument.                                  |
| round       | Number of round                                            |
| number      | Number of argument in the round                            |
| attacks     | Number of attacking argument from the previous round       |
| ann         | Type of argument (0-not an argument, 1-author, 2-reviewer) |
| text        | Text of argument\sentence                                  |

### Split dataset

You can use the dataset already split into train\val\test subsets. ([./dataset/sentence/](./dataset/sentence/))  
OR you can prepare a dataset by yourself using a script [./src/dataset/prepare.py](./src/dataset/prepare.py)

## Visualization

You can use save_annotated_text_html func from [./src/utils.py](./src/utils.py)

Example of result represented in [./assets/admsci5030125_boyarkin.html](./assets/admsci5030125_boyarkin.html):


![visualization_example.png](./assets/visualization_example.png)

## Statistics

Krippendorff's alpha for the dataset is _0.81±0.19_ [[link]](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha)

## Models

Available models:
- [Simple NN](./src/models/simple_model.py) 
- [RNN](./src/models/rnn_simple_model.py)
- [LSTM](./src/models/lstm_model.py)
- [BERT](./src/models/bert_model.py)


You can train models using [./src/model_training.py](./src/model_training.py) script. To choose a model you need to uncomment line with a desired model.
