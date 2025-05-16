# train_model.py
import torch
import torchmetrics
import matplotlib.pyplot as plt
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer, EsmModel

class ESMClassifier(LightningModule):
    def __init__(self, n_classes=25):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        self.embedder = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        d_model = self.embedder.config.hidden_size
        self.classifier = nn.Linear(d_model, n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes)

        self.train_acc_history = []
        self.val_acc_history = []

    def forward(self, x):
        encoded = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.embedder(**encoded)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_emb)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_accuracy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_accuracy(logits, y)
        f1 = self.val_f1(logits, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_acc_history.append(self.train_accuracy.compute().item())
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        self.val_acc_history.append(self.val_accuracy.compute().item())
        self.val_accuracy.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=2e-5)

    def plot_accuracy(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_acc_history, label="Train Acc")
        plt.plot(self.val_acc_history, label="Val Acc")
        plt.title("Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig("Accuracy Plot")
        plt.show()
