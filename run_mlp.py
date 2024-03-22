import os
import csv
import torch
import torchaudio
import yaml
from src.classifier import Classifier
from torchaudio.transforms import Resample
from statistics import mode

def load_model(model_file, config):
    model = Classifier(config)
    model.load_state_dict(torch.load(model_file))
    return model

def classify_audio(model_folder, path_to_audio_files, path_to_save, config_data, config_mlp):
    predictions = []

    # Load all sub models from model folder
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]
    models = [load_model(os.path.join(model_folder, model_file), config_mlp) for model_file in model_files]

    for file_name in os.listdir(path_to_audio_files):
        audio, sample_rate = torchaudio.load(os.path.join(path_to_audio_files, file_name))
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=config_data["desired_sample_rate"])
        single_batch = audio.unsqueeze(0)

        # Predict with all models
        model_predictions = []
        for model in models:
            x = model(single_batch)
            x = x.softmax(dim=1)
            x = x.squeeze()
            indices = x.argmax().item()
            model_predictions.append(indices)

        # Voting between sub models
        voting_prediction = mode(model_predictions)
        predicted_class = config_data["classes_reversed"][voting_prediction + 1]

        # Append filename and final predicted class to the predictions list
        predictions.append([file_name, predicted_class])

    # Call the method to write predictions to CSV
    write_predictions_to_csv(predictions, path_to_save)

        
def write_predictions_to_csv(predictions, path_to_save):
    # Create the "predictions" folder if it doesn't exist
    os.makedirs(path_to_save, exist_ok=True)

    # Find the next available CSV filename
    i = 0
    while True:
        i += 1
        csv_filename = os.path.join(path_to_save, f'predictions_{i}.csv')
        if not os.path.exists(csv_filename):
            break

    # Write the predictions to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['Filename', 'Predicted Class'])  # Add header
        csv_writer.writerows(predictions)

    print(f'Predictions saved to {csv_filename}')
    
    
if __name__ == '__main__':
    with open('config_mlp.yml', 'r') as file:
        config_mlp = yaml.safe_load(file)

    with open('config_data.yml', 'r') as file:
        config_data = yaml.safe_load(file)

    with open('config_run_mlp.yml', 'r') as file:
        config_run = yaml.safe_load(file)

    model_folder = config_run["model_file"]
    path_to_audio_files = config_run["path_to_audio_files"]
    path_to_save = config_run["path_to_save"]

    classify_audio(model_folder, path_to_audio_files, path_to_save, config_data, config_mlp)

