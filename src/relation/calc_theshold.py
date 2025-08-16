import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import f1_score
import torch
from transformers import AutoTokenizer
from itmo_pirsii_2023_diploma.src.relation.lstm_model import LSTMClassifier

# Disable tk warnings
matplotlib.use('Agg')

if __name__ == "__main__":
    tresholds = torch.linspace(0, 1, 1000)
    y_hat = torch.load("../../examples/y_hat", weights_only=False)
    y = torch.load("../../examples/y", weights_only=False)

    metric = []
    for tr in tresholds:
        metric.append(f1_score(y, y_hat>tr))

    metric = torch.tensor(metric)
    max_index = torch.argmax(metric)
    max_point_y = metric[max_index]
    max_point_x = tresholds[max_index]

    default_x = 0.5
    default_y = f1_score(y, y_hat>default_x)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=tresholds, y=metric)
    plt.annotate(f'Defalut: x={default_x:.2}, y={default_y:.4}',
                 xy=(default_x, default_y),
                 xytext=(default_x-0.1, default_y - 0.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12)
    plt.annotate(f'Max: x={max_point_x:.2}, y={max_point_y:.4}',
                 xy=(max_point_x, max_point_y),
                 xytext=(max_point_x, max_point_y - 0.2),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12)

    plt.xlabel("threshold")
    plt.ylabel("F-1 score")
    plt.savefig("../../assets/f1-threshold-relation.png")  # 0.8689

