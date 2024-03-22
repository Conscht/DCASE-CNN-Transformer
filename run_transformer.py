import os
import csv
import torchaudio
import yaml
from collections import Counter
from torchaudio import transforms
from src.transformer_module import TranformerModule
from tqdm import tqdm


def main():
    with open('config_transformer.yml', 'r') as file:
        config = yaml.safe_load(file)

    classify_audio(config)

def load_models(config):
    model_paths = config['run']['model_paths']  # Annahme: model_paths ist eine Liste von Pfaden
    models = []
    for path in model_paths:
        model = TranformerModule.load_from_checkpoint(path)
        model.eval()
        model.to('cpu')
        models.append(model)
    return models

def classify_audio(config):
    predictions = []
    models = load_models(config)
    path_to_files = config["run"]["path_to_files"]

    # Normalization parameters
    mean = config["data"]["mean"] 
    std = config["data"]["std"]  
    
    mel = transforms.MelSpectrogram(sample_rate=config["data"]["desired_sample_rate"], n_fft=640, hop_length=320, n_mels=40)
    logmel = transforms.AmplitudeToDB()
    
    for file_name in tqdm(os.listdir(path_to_files), desc="Classifying"):
        votes = []
        audio, sample_rate = torchaudio.load(os.path.join(path_to_files, file_name))
        if audio.shape[0] > 1: audio = audio.mean(dim=0)
        audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=config["data"]["desired_sample_rate"])
        
        audio = mel(audio)
        audio = logmel(audio)
        
        # Normalize audio
        audio = (audio - mean) / std
        
        audio = audio.permute(1, 0)
        single_batch = audio.unsqueeze(0)
        
        for model in models:
            x = model(single_batch)
            x = x.softmax(dim=1)
            x = x.squeeze()
            index = x.argmax().item()
            predicted_class = config["data"]["classes_reversed"][index + 1]
            votes.append(predicted_class)

        # Voting mechanism
        vote_result = Counter(votes)
        final_prediction = vote_result.most_common(1)[0][0]
        predictions.append([file_name, final_prediction])

        # Print the number of unique votes
        unique_votes = len(vote_result)
        print(f"File: {file_name}, Unique Votes: {unique_votes}, Prediction: {final_prediction}")

    write_predictions_to_csv(predictions, config["run"]["path_to_save"])

def write_predictions_to_csv(predictions, path_to_save):
    os.makedirs(path_to_save, exist_ok=True)
    csv_filename = os.path.join(path_to_save, 'predictions.csv')  # Using a single file for simplicity

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['Filename', 'Predicted Class'])
        csv_writer.writerows(predictions)

    # Use the length of the predictions list to report the number of files processed
    print(f'\nTotal of {len(predictions)} files were processed and predictions saved to {csv_filename}')
    
if __name__ == '__main__':
    main()
