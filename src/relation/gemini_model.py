import os

import numpy as np
import pytorch_lightning as pl
from google import genai
from google.genai.types import GenerateContentConfig
from ratelimit import limits, sleep_and_retry
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

import logging

class GeminiClassifierModel(pl.LightningModule):
    def __init__(self, loss_type: str, model_name: str, prompt_dir: str, prompt_id: int | str, **kwargs):
        super(GeminiClassifierModel, self).__init__()
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#generationconfig
        self.generation_config = GenerateContentConfig(top_k=1, top_p=0, temperature=0, seed=0)
        self.prompt_text = self.read_prompt(prompt_dir, prompt_id)
        self.loss = self.get_loss(loss_type)

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
        if loss_type == "binary_cross_entropy":
            return nn.BCELoss()
        raise AttributeError(f"Loss {loss_type} not found")

    def read_prompt(self, dir: str, prompt_id: int):
        with open(os.path.join(dir, str(prompt_id) + ".txt"), "r") as file:
            text = file.read()
        return text

    def get_prompt(self, sentences_1, sentences_2):
        prompt = [self.prompt_text]
        for idx in range(len(sentences_1)):
            # Added ":" at the end 26/03/25
            prompt.append(f"Pair {idx+1}:")
            prompt.append(sentences_1[idx])
            prompt.append(sentences_2[idx])
        return prompt

    def validate_response(self, response, num_sent):
        parsed_response = list(map(int, response))
        if num_sent != len(parsed_response):
            logging.info("Response len (%s) <> num_sent (%s)", len(parsed_response), num_sent)
            parsed_response.extend([-1 for _ in range(abs(num_sent-len(parsed_response)))])
        return parsed_response

    def forward(self, sentences_1, sentences_2):
        prompt = self.get_prompt(sentences_1, sentences_2)
        response = self.api_call(prompt)
        parsed_response = response.strip().split(",")
        parsed_response = self.validate_response(parsed_response, len(sentences_1))
        return parsed_response

    @sleep_and_retry
    @limits(29, 65)
    def api_call(self, prompt):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.generation_config,
        )
        return response.text

    def test_step(self, batch):
        sent, y = batch
        y_hat = self.forward(*sent)
        loss = self.loss(y_hat, y)

        y_true = self.get_logit(y.cpu().detach().numpy())
        y_pred_logit = self.get_logit(y_hat.cpu().detach().numpy())

        self.test_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred_logit})
        return y_pred_logit

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

        # y_pred = (y_pred > self.threshold).astype(np.float32)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        if epoch_type == "test":
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
            self.log(f"epoch/f1_score/{epoch_type}", f1)
            # self.log(f"epoch/error_rate/{epoch_type}", error_rate)

        if epoch_type in ("test", "val"):
            # Plot confusion matrix
            self.plot_confusion_matrix(y_true, y_pred, epoch_type)

        step_outputs.clear()  # free memory
