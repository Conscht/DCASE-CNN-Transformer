import torch
import yaml
from src.dataset import AudioDataset
from transformer_module import TranformerModule
from torch.utils.data import DataLoader

# Konfigurationen (als Beispiel)
with open('config_transformer.yml', 'r') as file:
    config = yaml.safe_load(file)

# Initialisiere Modell und Dataset
model = TranformerModule(config)
dataset = AudioDataset(config["data"]["path_to_file_names"] + f"fold{1}_train.txt", 
                        config["data"]['path_to_files'], config["data"]['classes'], desired_sample_rate=config["data"]['desired_sample_rate'])
loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

# Teste einen Batch
for batch in loader:
    x, y = batch
    print("Input shape:", x.shape)
    print("Label shape:", y.shape)

    # Teste das Modell
    y_hat = model(x)
    print("Output shape:", y_hat.shape)

    # Teste den Verlust und die Genauigkeit
    loss = model.training_step(batch, 0)
    print("Loss:", loss)

    break  # Nur ein Batch für den Test
