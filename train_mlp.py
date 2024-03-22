from argparse import ArgumentParser
import yaml
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch.utils.data import DataLoader, random_split

from src.classifier import Classifier
from src.dataset import AudioDataset

# Trade off accuracy for performance 
torch.set_float32_matmul_precision('high')

def train(config_mlp, config_data):
    pl.seed_everything(1234)

    # ------------
    # Models
    # ------------
    models = []
    print("Creating models...")
    for fold in range(1, 5):
        model = Classifier(config_mlp)
        models.append(model)
    
    for fold in range(1, 5):
        # ------------
        # Data
        # ------------
        print(f"data sets (fold {fold})...")
        dataset_train = AudioDataset(config_data["fold_path"] + f"fold{fold}_train.txt", 
                                    config_data['path_to_audio_files'], config_data['classes'], desired_sample_rate=config_data['desired_sample_rate'])
        dataset_evaluate = AudioDataset(config_data["fold_path"] + f"fold{fold}_evaluate.txt",
                                        config_data['path_to_audio_files'], config_data['classes'], desired_sample_rate=config_data['desired_sample_rate'])
        dataset_test = AudioDataset(config_data["fold_path"] + f"fold{fold}_test.txt", config_data['path_to_audio_files'], config_data['classes'], desired_sample_rate=config_data['desired_sample_rate'], 
                                    has_labels=False)

        print(f"Building data loaders (fold {fold})...")
        train_loader = DataLoader(dataset_train, batch_size=config_mlp['batch_size'], shuffle=True, num_workers=11)
        val_loader = DataLoader(dataset_evaluate, batch_size=config_mlp['batch_size'], num_workers=11)
        test_loader = DataLoader(dataset_test, batch_size=config_mlp['batch_size'], num_workers=11)


        # ------------
        # Training
        # ------------
        print(f"Training model (fold {fold})...")  
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=config_mlp['early_stopping_patience'], mode='min')
        checkpoint_callback = ModelCheckpoint(
                                dirpath='checkpoints/',
                                filename=f'mlp-fold{fold}' + '{epoch}-{val_loss:.2f}',
                                save_top_k=3,
                                verbose=True,
                                monitor='val_loss',
                                mode='min'
                            )

        trainer = pl.Trainer(
            max_epochs=config_mlp['max_epochs'],
            min_epochs=config_mlp['min_epochs'],
            accelerator="auto",
            callbacks=[early_stopping_callback, checkpoint_callback]
        )

        trainer.fit(model, train_loader, val_loader)

        

    # ------------
    # Saving
    # ------------
    model_folder_base = config_mlp['model_folder']

    # Find smallest model number 
    smallest_model_number = 1
    while True:
        model_folder = os.path.join(model_folder_base, f"model_{smallest_model_number}")

        if not os.path.exists(model_folder):
            break

        smallest_model_number += 1

    # Create model folder
    os.makedirs(model_folder)

    for fold in range(1, 5):
        model_filename = f"model_fold_{fold}.pth"
        model_filepath = os.path.join(model_folder, model_filename)

        # Saving the individual model folds
        print(f"Saving model for fold {fold}...")
        torch.save(model.state_dict(), model_filepath)
        print(f"Model saved to {model_filepath}.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_mlp', type=str, default='config_mlp.yml', help='Path to the configuration file')
    parser.add_argument('--config_data', type=str, default='config_data.yml', help='Path to the data configuration file')
    args = parser.parse_args()

    with open(args.config_mlp, 'r') as file:
        config_mlp = yaml.safe_load(file)

    with open(args.config_data, 'r') as file:
        config_data = yaml.safe_load(file)

    train(config_mlp, config_data)
