import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from src.transformer import TransformerEncoderOnly  
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

class TranformerModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = TransformerEncoderOnly(
            input_dim=config['model']['input_dim'], 
            num_classes=config['model']['num_classes'],
            d_model=config['model']['d_model'],
            nheads=config['model']['nheads'],
            num_layers=config['model']['num_layers'],
            d_ff=config['model']['d_ff'],
            dropout=config['model']['dropout'],
            max_length=config['model']['max_length'] 
        )
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=config['model']['num_classes'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams['training']['learning_rate'],
            weight_decay=self.hparams["training"]["weight_decay"]
        )

        scheduler = self.hparams['training']['lr_scheduler']

        if scheduler == 'plateau':
            scheduler = {
                'scheduler': ReduceLROnPlateau(
                    optimizer, 
                    factor=self.hparams['training']['plateau_reduce_factor'], 
                    patience=self.hparams['training']['plateau_reduce_patience'], 
                    mode='min'
                ),
                'monitor': 'val_loss' 
            }
        elif scheduler == 'cos':
            scheduler = {
                'scheduler': CosineAnnealingLR(
                    optimizer, 
                    T_max=self.hparams['training']['T_max'], 
                    eta_min=self.hparams['training']['eta_min']  
                ),
                'interval': 'epoch'
            }
        else:
            return optimizer

        return [optimizer], [scheduler]




