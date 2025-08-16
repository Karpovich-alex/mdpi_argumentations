import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification

from itmo_pirsii_2023_diploma.src.models.base_model import BaseClassifierModel


class BERTClassifier(BaseClassifierModel):
    def __init__(self, tokenizer_name, hidden_dim: list[int], output_dim, loss_type, optimizer, dropout_fc,
                 **kwargs):
        super(BERTClassifier, self).__init__(loss_type, optimizer, output_dim)
        self.save_hyperparameters()

        self.input_sent_num = 2

        self.bert_model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, num_labels=output_dim)

        if not isinstance(hidden_dim, list):
            if isinstance(hidden_dim, int):
                hidden_dim = [hidden_dim]
            else:
                raise AttributeError("hidden_dims must be int or list[int]. Find:", type(hidden_dim))
        hidden_dim.insert(0, self.bert_model.config.dim)
        hidden_dim.append(self.output_dim)
        self.hidden_dim = hidden_dim[0]

        self.optimizer_params = optimizer

        # self.fc = self.get_fc_layer(hidden_dim)
        # self.dropout = nn.Dropout(p=dropout_fc)
        # self.softmax = nn.Softmax(dim=1)

    def get_fc_layer(self, hidden_dims):
        fc = nn.Sequential()
        for i in range(len(hidden_dims) - 1):
            input_size = hidden_dims[i]
            output_size = hidden_dims[i + 1]

            fc.append(nn.Linear(input_size, output_size))
            if i != len(hidden_dims) - 2:
                fc.append(nn.Sigmoid())
        return fc

    def forward(self, input_ids, attention_mask):

        x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        # x = x[:, 0, :]  # Выбираем только [CLS]
        # x = self.fc(x)
        # x = self.dropout(x).logits
        # x = self.softmax(x)
        return torch.clamp(x.logits, 0, 1)
