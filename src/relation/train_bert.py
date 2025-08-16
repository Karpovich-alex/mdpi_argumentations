import pandas as pd
import pytorch_lightning as pl
from transformers import AutoTokenizer

from itmo_pirsii_2023_diploma.src.relation.bert_model import BERTClassifier

# Disable tk warnings
import matplotlib
matplotlib.use('Agg')

def get_lstm_model(params):
    return BERTClassifier(**params), "bert"


if __name__ == "__main__":
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import EarlyStopping

    from loader import get_loaders

    use_clearml = True  # True = use clearML, False = dont use
    continue_last_model = False  # True = use last model, need to set ckpt_path in section below, False = train model from zero

    if use_clearml:
        from clearml import Task

        project_name = "Diploma"
        task_tag = "relation"
        task = Task.init(project_name=project_name, tags=[task_tag],
                         continue_last_task=continue_last_model,
                         )  # , continue_last_task=True, reuse_last_task_id='283151c414814d9793e80284e2b8d1d7'

    # Set params
    max_length = 256
    tokenizer_name = "distilbert-base-uncased"
    dataset_version = "v4"
    # Set seed
    seed = 0

    params = {
        "seed": seed,
        "tokenizer_name": tokenizer_name,
        "max_epochs": 100,
        "batch_size": 64,
        "hidden_dim": [768],
        "output_dim": 1,
        "dropout_fc": 0,
        "max_length": max_length,
        "optimizer": {"name": "AdamW",
                      "lr": 1e-4},
        "loss_type": "binary_cross_entropy",
        "dataset_version": dataset_version,
        "debug_size": 10
    }

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)

    params["vocab_size"] = vocab_size

    if use_clearml: task.connect(params)

    pl.seed_everything(seed)

    # Load data
    train_loader, val_loader, test_loader, debug_loader = get_loaders(
        f"../../dataset/relation/train.tsv",
        f"../../dataset/relation/val.tsv",
        f"../../dataset/relation/test.tsv",
        tokenizer=tokenizer,
        batch_size=params["batch_size"],
        max_length=max_length,
        debug_size=params["debug_size"],
        use_bert=True, output_dim=params["output_dim"])

    # Get model
    model, tag = get_lstm_model(params)

    if use_clearml:
        task.set_model_config(config_text=str(model))
        task.set_name(model.name)
        task.add_tags(tag)
        task.add_tags(f"ds:{params['dataset_version']}")

    # Configure trainer
    early_stopping = EarlyStopping(monitor="epoch/loss/val", min_delta=0.001, patience=5, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=params["max_epochs"],
                         logger=TensorBoardLogger("F:/Projects/diplom/tb_logs", name=model.name),
                         callbacks=[early_stopping],
                         )  # callbacks=[early_stopping]

    # Train and test model
    if not continue_last_model:
        trainer.fit(model, train_loader,
                    val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_dataloaders=val_loader,
                    ckpt_path=r"E:/Projects/diplom/tb_logs\LSTMClassifier\version_94\checkpoints\epoch=68-step=4968.ckpt")  # , ckpt_path=r"<path-to-ckpt>"

    trainer.test(model, test_loader, ckpt_path="best")

    # model = LSTMClassifier.load_from_checkpoint(r"E:/Projects/diplom/tb_logs\LSTMClassifier\version_71\checkpoints\epoch=36-step=2664.ckpt",).to("cuda")

    if debug_loader is not None:
        model.eval()
        debug_data = []
        for idx, data in enumerate(debug_loader):
            if data[0][0].device.type != model.device:
                for i in range(2):
                    data[i] = data[i].to(model.device)
            pred = model(*data[:-1])
            debug_data.append({
                "id": idx,
                "predicted": pred[0].tolist(),
                "actual": data[-1].tolist()
            })
        task.get_logger().report_table(title="debug", series=f"Samples dataset: {dataset_version} epoch: {trainer.current_epoch}", table_plot=pd.DataFrame(debug_data))

    if use_clearml:
        task.close()