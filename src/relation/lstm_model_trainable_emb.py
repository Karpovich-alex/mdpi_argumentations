import torch
import torch.nn as nn

from itmo_pirsii_2023_diploma.src.models.base_model import BaseClassifierModel


class LSTMClassifier(BaseClassifierModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: list[int], n_layers, output_dim, loss_type, optimizer,
                 bidirectional, dropout, dropout_fc=0, **kwargs):
        super(LSTMClassifier, self).__init__(loss_type, optimizer, output_dim)
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        if not isinstance(hidden_dim, list):
            if isinstance(hidden_dim, int):
                hidden_dim = [hidden_dim]
            else:
                raise AttributeError("hidden_dims must be int or list[int]. Find:", type(hidden_dim))
        self.hidden_dim = hidden_dim[0]
        self.n_layers = n_layers
        self.optimizer_params = optimizer

        self.embedding1 = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding2 = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, bidirectional=bidirectional,
                            dropout=dropout)
        self.sigmoid = nn.Sigmoid()
        self.fc = self.get_fc_layer(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_fc)
        self.softmax = nn.Softmax(dim=1)

    def get_fc_layer(self, hidden_dims):
        fc = nn.Sequential()
        for i in range(len(hidden_dims)):
            if i == 0:
                input_size = self.hidden_dim * 2
            else:
                input_size = hidden_dims[i]

            if i == len(hidden_dims)-1:
                output_size = self.output_dim
            else:
                output_size = hidden_dims[i + 1]

            fc.append(nn.Linear(input_size, output_size))
            if i != len(hidden_dims) - 1:
                fc.append(nn.Sigmoid())
        return fc

    def forward(self, input_ids, attention_mask):
        lstm_result = []
        for i in range(2):
            embedded = getattr(self, f"embedding{i + 1}")(input_ids[i])
            packed_embedded = (
                nn.utils.rnn.pack_padded_sequence(embedded, attention_mask[i].sum(1).cpu(), batch_first=True,
                                                  enforce_sorted=False))

            outputs, (hidden, cell) = self.lstm(packed_embedded)
            lstm_result.append(hidden[0])
        x = torch.concatenate(lstm_result, dim=1)
        x = self.sigmoid(x)
        x = self.fc(x)
        # x = self.dropout(x)
        # x = self.softmax(x)
        # x = self.sigmoid(x)
        return x
