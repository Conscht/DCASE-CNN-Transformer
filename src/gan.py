import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Definiere den Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim):
        super().__init__()
        # Anpassen für deine Architektur
        self.net = nn.Sequential(
            # Beispielarchitektur
            nn.Linear(noise_dim + label_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Kombiniere Noise und Labels
        x = torch.cat([noise, labels], 1)
        return self.net(x)

# Definiere den Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super().__init__()
        # Anpassen für deine Architektur
        self.net = nn.Sequential(
            # Beispielarchitektur
            nn.Linear(input_dim + label_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # Kombiniere X und Labels
        x = torch.cat([x, labels], 1)
        return self.net(x)
