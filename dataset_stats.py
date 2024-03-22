import torch
import torchaudio
from torchaudio import transforms
import os
from tqdm import tqdm

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_files, classes, sample_rate):
        self.path_to_files = path_to_files
        self.classes = classes
        self.sample_rate = sample_rate
        self.mel_transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=640, hop_length=320, n_mels=40)
        self.db_transform = transforms.AmplitudeToDB()
        self.file_paths = [os.path.join(path_to_files, f) for f in os.listdir(path_to_files) if f.endswith('.wav')]

    def __getitem__(self, index):
        audio_path = self.file_paths[index]
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.mean(dim=0) if audio.shape[0] > 1 else audio
        audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=self.sample_rate)
        audio = self.mel_transform(audio)
        audio = self.db_transform(audio)
        return audio

    def __len__(self):
        return len(self.file_paths)

def calculate_dataset_statistics(dataset):
    all_spectrograms = []
    for i in tqdm(range(len(dataset)), desc="Calculating statistics"):
        audio = dataset[i]
        all_spectrograms.append(audio)
    
    all_spectrograms = torch.stack(all_spectrograms)
    mean = all_spectrograms.mean()
    std = all_spectrograms.std()
    return mean.item(), std.item()

if __name__ == "__main__":
    # Update these paths and parameters according to your dataset
    path_to_files = "/data/baproj/dlap/TUT-acoustic-scenes-2017-development/audio/"
    classes = {'beach': 1, 'bus': 2}  # Update this with your actual classes
    sample_rate = 44100
    
    dataset = AudioDataset(path_to_files=path_to_files, classes=classes, sample_rate=sample_rate)
    
    mean, std = calculate_dataset_statistics(dataset)
    print(f"Calculated Mean: {mean}, Standard Deviation: {std}")
