import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from src.gan import Generator, Discriminator

class CGAN(pl.LightningModule):
    def __init__(self, noise_dim, label_dim, output_dim, lr=0.0002):
        super().__init__()
        self.generator = Generator(noise_dim, label_dim, output_dim)
        self.discriminator = Discriminator(output_dim, label_dim)
        self.noise_dim = noise_dim
        self.lr = lr

    def forward(self, noise, labels):
        return self.generator(noise, labels)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, labels = batch

        # Implementiere den Trainingszyklus für Generator und Discriminator
        # Hier musst du die Loss-Funktionen definieren und die Optimierungsschritte durchführen

        pass

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [opt_g, opt_d], []