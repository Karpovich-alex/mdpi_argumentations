from src.models.base_model import BaseClassifierModel
from transformers import BertForSequenceClassification


class BERTClassifier(BaseClassifierModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, loss_type, optimizer, model_name,
                 **kwargs):
        super(BERTClassifier, self).__init__(loss_type, optimizer)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.model = BertForSequenceClassification.from_pretrained(model_name,
                                                                   num_labels=output_dim,
                                                                   output_attentions=True,
                                                                   output_hidden_states=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        logits = outputs.logits
        return logits
