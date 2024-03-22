import os
from datetime import datetime
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.delayed_early_stopping import DelayedEarlyStopping
from src.dataset import AudioDataset
from torch.utils.data import DataLoader
from src.transformer_module import TranformerModule

def main():
    pl.seed_everything(1233)
    torch.set_float32_matmul_precision("high")

    config = load_config("config_transformer.yml")
    base_name = generate_base_name(config['training']['log_name'])

    for fold in range(2, config["training"]["folds"] + 1):
        print(f"Processing fold {fold}...")
        train_loader, val_loader = create_dataloaders(config, fold)
        model = create_model(config)

        logger = setup_logger(config["training"]["lightning_log_folder"], base_name, fold)
        callbacks = setup_callbacks(config["training"], base_name, fold)

        model = load_model_if_checkpoint_exists(config['training'], model, fold)

        trainer = Trainer(logger=logger, max_epochs=config['training']['max_epochs'],
                          min_epochs=config['training']['min_epochs'], accelerator="auto",
                          callbacks=callbacks)

        trainer.fit(model, train_loader, val_loader)
        save_model(model, config["training"]["model_folder"], base_name, fold)

def load_config(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)

def generate_base_name(log_name):
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    return f"{log_name}-{timestamp}"

def setup_logger(log_folder, base_name, fold):
    unique_name = f"{base_name}-fold{fold}"
    return TensorBoardLogger(save_dir=log_folder, name=unique_name)

def create_model(config):
    return TranformerModule(config)

def create_dataloaders(config, fold):
    train_file_names_path = f"{config['data']['path_to_file_names']}fold{fold}_train.txt"
    val_file_names_path = f"{config['data']['path_to_file_names']}fold{fold}_evaluate.txt"
    mean = config["data"]["mean"]
    std = config["data"]["std"]

    train_dataset = AudioDataset(
        path_to_file_names=train_file_names_path,
        path_to_files=config['data']['path_to_files'],
        mean=mean,
        std=std,
        classes=config['data']['classes'],
        sample_rate=config['data']['desired_sample_rate'],
        augmentation_chance=config['data']['augmentation_chance'],
        noise_chance=config['data']['noise_chance'],
        aug_strength=config['data']['aug_strength'],
        freq_mask_param=config['data']['freq_mask_param'],
        time_mask_param=config['data']['time_mask_param']
    )
    
    val_dataset = AudioDataset(
        path_to_file_names=val_file_names_path,
        path_to_files=config['data']['path_to_files'],
        mean=mean,
        std=std,
        classes=config['data']['classes'],
        sample_rate=config['data']['desired_sample_rate'],
    )
    
    # DataLoader für Trainings- und Validierungsdatensätze
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=config["training"].get("num_workers", 11) 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=config["training"].get("num_workers", 11)
    )
    
    return train_loader, val_loader


def setup_callbacks(training_config, base_name, fold):
    checkpoint_filename = f"{base_name}-fold{fold}-{{epoch}}-{{val_loss:.2f}}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=training_config["checkpoint_folder"],
        filename=checkpoint_filename,
        save_top_k=training_config["save_top_k"],
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    early_stopping_start = training_config.get("early_stopping_start", 0)
    early_stopping_callback = DelayedEarlyStopping(
        start_epoch=early_stopping_start,
        monitor='val_loss',
        patience=training_config["early_stopping_patience"],
        verbose=True,
        mode='min'
    )

    return [checkpoint_callback, early_stopping_callback]


def load_model_if_checkpoint_exists(training_config, model, fold):
    if training_config.get('resume_from_checkpoint', False):
        checkpoint_path = training_config.get('checkpoint_path', None)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            return TranformerModule.load_from_checkpoint(checkpoint_path, config=training_config)
        else:
            print("Checkpoint path not provided or does not exist.")
    return model

def save_model(model, model_folder, base_name, fold):
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_filename = f"{base_name}-fold{fold}.pth"
    model_filepath = os.path.join(model_folder, model_filename)
    torch.save(model.state_dict(), model_filepath)
    print(f"Model saved to {model_filepath}.")

if __name__ == "__main__":
    main()
