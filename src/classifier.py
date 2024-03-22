import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from src.model import SimpleModel
import torchmetrics

class Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.backbone = SimpleModel(hidden_dim=128)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=config['num_classes'])

    def forward(self, x):
        result = self.backbone(x)
        return result

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        accuracy = self.accuracy(y_hat, y)
        loss = F.cross_entropy(y_hat, y)
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy}, 
                      on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        accuracy = self.accuracy(y_hat, y)
        loss = F.cross_entropy(y_hat, y)
        self.log_dict({"val_loss": loss, "val_accuracy": accuracy}, 
                      on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])