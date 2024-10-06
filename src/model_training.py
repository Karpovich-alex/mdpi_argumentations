import pytorch_lightning as pl


def get_simple_model(batch_size, vocab_size, max_length, max_epochs, **kwargs):
    from models.simple_model import SimpleClassifier
    params = {
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "embedding_dim": 350,
        "hidden_dim": 256,
        "output_dim": 2,
        "n_layers": 1,
        "vocab_size": vocab_size,
        "max_length": max_length,
        "optimizer": {"name": "Adam",
                      "lr": 1e-4},
        "loss_type": "cross_entropy",
    }
    params.update(kwargs)
    return SimpleClassifier(**params), "simple"


def get_rnn_model(batch_size, vocab_size, max_length, max_epochs, **kwargs):
    from models.rnn_simple_model import RNNClassifier
    params = {
        "vocab_size": vocab_size,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "embedding_dim": 350,
        "hidden_dim": 512,
        "output_dim": 2,
        "n_layers": 1,
        "max_length": max_length,
        "optimizer": {"name": "Adam",
                      "lr": 2e-3},
        "loss_type": "cross_entropy",
    }
    params.update(kwargs)
    return RNNClassifier(**params), "rnn"


def get_lstm_model(batch_size, vocab_size, max_length, max_epochs, **kwargs):
    from models.lstm_model import LSTMClassifier
    params = {
        "vocab_size": vocab_size,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "embedding_dim": 350,
        "hidden_dim": 18,
        "hidden_dim2": 12,
        "output_dim": 2,
        "n_layers": 1,
        "bidirectional": False,
        "dropout": 0.75,
        "max_length": max_length,
        "optimizer": {"name": "Adam",
                      "lr": 6e-4},
        "loss_type": "cross_entropy",
        "dataset_version": "v2"
    }
    params.update(kwargs)
    return LSTMClassifier(**params), "lstm"


def get_bert_model(batch_size, vocab_size, max_length, max_epochs, **kwargs):
    from models.bert_model import BERTClassifier
    params = {
        "vocab_size": vocab_size,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "embedding_dim": 350,
        "hidden_dim": 512,
        "output_dim": 2,
        "n_layers": 1,
        # "max_length": max_length,
        "optimizer": {"name": "AdamW",
                      "lr": 2e-5},
        "loss_type": "cross_entropy",
        "model_name": kwargs.get("model_name", "bert-base-uncased"),
    }
    params.update(kwargs)
    return BERTClassifier(**params), "bert"


if __name__ == "__main__":
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint

    from models.base_model import get_loaders

    use_clearml = True  # True = use clearML, False = dont use
    continue_last_model = False  # True = use last model, need to set ckpt_path in section below, False = train model from zero

    if use_clearml:
        from clearml import Task
        project_name = "MDPI"

    # Set seed
    seed = 0
    pl.seed_everything(seed)

    # Set params
    max_length = 110
    batch_size = 64
    max_epochs = 100
    tokenizer_name = "distilbert-base-uncased"

    # Load data
    train_loader, val_loader, test_loader, vocab_size = get_loaders("../dataset/sentence/train.csv",
                                                                    "../dataset/sentence/val.csv",
                                                                    "../dataset/sentence/test.csv",
                                                                    tokenizer_name=tokenizer_name,
                                                                    batch_size=batch_size,
                                                                    max_length=max_length)

    # Get model
    model, tag = get_rnn_model(batch_size, vocab_size, max_length, max_epochs, tokenizer_name=tokenizer_name)
    # model, tag = get_lstm_model(batch_size, vocab_size, max_length, max_epochs, tokenizer_name=tokenizer_name)
    # model, tag = get_bert_model(batch_size, vocab_size, max_length, max_epochs, model_name=tokenizer_name)

    # Configure trainer
    trainer = pl.Trainer(max_epochs=max_epochs, logger=TensorBoardLogger("./tb_logs", name=model.name))
    if use_clearml:
        task = Task.init(project_name=project_name, task_name=model.name, tags=[tag],
                         continue_last_task=continue_last_model)  # , continue_last_task=True
        task.set_model_config(config_text=str(model))

    # Train and test model
    if not continue_last_model:
        trainer.fit(model, train_loader,
                    val_dataloaders=val_loader)  # , ckpt_path=r"<path-to-ckpt>"
    else:
        trainer.fit(model, train_loader, val_dataloaders=val_loader)  # , ckpt_path=r"<path-to-ckpt>"
    trainer.test(model, test_loader, ckpt_path="best")

    if use_clearml:
        task.close()
