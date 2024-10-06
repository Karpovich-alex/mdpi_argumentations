import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from itmo_pirsii_2023_diploma.src.models.base_model import BaseClassifierModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class LSTMClassifier(BaseClassifierModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, loss_type, optimizer, bidirectional,
                 dropout, hidden_dim2, **kwargs):  # hidden_dim2
        super(LSTMClassifier, self).__init__(loss_type, optimizer)
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.optimizer_params = optimizer

        self.loss = self.get_loss(loss_type)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, bidirectional=bidirectional,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(self.hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

        self.training_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, attention_mask.sum(1).cpu(), batch_first=True,
                                                            enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed_embedded)
        # unpacked_outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        x = hidden[0]
        x = self.dropout(x)
        x = self.fc1(self.sigmoid(x))
        x = self.fc2(self.sigmoid(x))
        x = self.dropout(x)
        x = self.softmax(x)
        # result = self.dropout(result)
        return x
