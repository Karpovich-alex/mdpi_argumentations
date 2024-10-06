import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from itmo_pirsii_2023_diploma.src.dataset import SentenceDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer


class BaseClassifierModel(pl.LightningModule):
    def __init__(self, loss_type, optimizer, **kwargs):
        super(BaseClassifierModel, self).__init__()
        self.save_hyperparameters()

        self.optimizer_params = optimizer
        self.loss = self.get_loss(loss_type)

        self.training_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_outputs = []

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def get_loss(loss_type):
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        if loss_type == "mse_loss":
            return nn.MSELoss()
        if loss_type == "nll_loss":
            return nn.NLLLoss()
        raise AttributeError(f"Loss {loss_type} not found")

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        y_hat = self.forward(input_ids, attention_mask)
        loss = self.loss(y_hat, y)
        self.log("batch/train/loss", loss, prog_bar=False)

        y_true = y.argmax(axis=1).cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        acc = accuracy_score(y_true, y_pred)
        self.log("batch/train/acc", acc)

        self.training_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})
        return loss

    def on_train_epoch_end(self):
        self.on_epoch_end(epoch_type="train")

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        y_hat = self.forward(input_ids, attention_mask)
        loss = self.loss(y_hat, y)
        # self.log("batch/val/loss", loss, prog_bar=False)

        y_true = y.argmax(axis=1).cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        # acc = accuracy_score(y_true, y_pred)
        # self.log("batch/val/acc", acc)

        self.validation_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})
        return loss

    def on_validation_epoch_end(self):
        self.on_epoch_end(epoch_type="val")

    def test_step(self, batch):
        input_ids, attention_mask, y = batch
        y_hat = self.forward(input_ids, attention_mask)
        loss = self.loss(y_hat, y)

        y_true = y.argmax(axis=1).cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        self.test_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})
        return y_pred

    def on_test_epoch_end(self):
        self.on_epoch_end(epoch_type="test")

    def on_epoch_end(self, epoch_type="train"):
        if epoch_type == "val":
            step_outputs = self.validation_step_outputs
        elif epoch_type == "train":
            step_outputs = self.training_step_outputs
        elif epoch_type == "test":
            step_outputs = self.test_step_outputs
        else:
            raise ValueError("Cant understand epoch_type %s", epoch_type)

        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])

        for results_dict in step_outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        # error_rate = 1 - acc
        if epoch_type == "test":
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            self.log(f"{epoch_type}/loss", loss.mean())
            self.log(f"{epoch_type}/acc", acc)
            # self.log(f"{epoch_type}/error_rate", error_rate)
            self.log(f"{epoch_type}/f1_score", f1)
            self.log(f"{epoch_type}/precision", precision)
            self.log(f"{epoch_type}/recall", recall)
        else:
            self.log(f"epoch/loss/{epoch_type}", loss.mean())
            self.log(f"epoch/acc/{epoch_type}", acc)
            # self.log(f"epoch/error_rate/{epoch_type}", error_rate)
        step_outputs.clear()  # free memory

    def configure_optimizers(self):
        params = self.optimizer_params.copy()
        optimizer_name = params.pop("name")
        if optimizer_name == "SGD":
            return torch.optim.SGD(self.parameters(), **params)
        if optimizer_name == "Adam":
            return torch.optim.Adam(self.parameters(), **params)
        if optimizer_name == "Adagrad":
            return torch.optim.Adagrad(self.parameters(), **params)
        if optimizer_name == "AdamW":
            return torch.optim.AdamW(self.parameters(), **params)
        raise AttributeError(f"Optimizer {optimizer_name} not found")
