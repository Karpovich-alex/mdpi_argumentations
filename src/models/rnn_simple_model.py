import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from itmo_pirsii_2023_diploma.src.models.base_model import BaseClassifierModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class RNNClassifier(BaseClassifierModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, loss_type, optimizer, **kwargs):
        super(RNNClassifier, self).__init__(loss_type, optimizer)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, self.n_layers)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        embedded = self.embedding(input_ids)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, attention_mask.sum(1).cpu(), batch_first=True,
                                                            enforce_sorted=False)
        outputs, _ = self.rnn(packed_embedded)
        unpacked_outputs, lens_unpacked = nn.utils.rnn.pad_packed_sequence(outputs,
                                                                           batch_first=True)  # B, T, hidden_dim
        last_outputs = unpacked_outputs[torch.arange(batch_size), lens_unpacked - 1]
        result = self.fc(self.sigmoid(last_outputs))
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids, attention_mask, _ = batch
        return self(input_ids, attention_mask)
