from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer

from src.dataset.dataset import SentenceDataset


def get_loaders(train_path: str, val_path: str, test_path: str, tokenizer_name='distilbert-base-uncased', batch_size=32,
                max_length=350, num_workers=7):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    if train_path.endswith('.json'):
        train_dataset = SentenceDataset.from_json(train_path, tokenizer, max_length)
        val_dataset = SentenceDataset.from_json(val_path, tokenizer, max_length)
        test_dataset = SentenceDataset.from_json(test_path, tokenizer, max_length)
    else:
        train_dataset = SentenceDataset.from_csv(train_path, tokenizer, max_length)
        val_dataset = SentenceDataset.from_csv(val_path, tokenizer, max_length)
        test_dataset = SentenceDataset.from_csv(test_path, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             persistent_workers=True)
    return train_loader, val_loader, test_loader, vocab_size
