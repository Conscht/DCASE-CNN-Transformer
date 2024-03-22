import random
import re
import torch
import torchaudio
from torchaudio import transforms

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_file_names, path_to_files, classes, sample_rate, mean, std,
                 augmentation_chance=0, has_labels=True, noise_chance=0, aug_strength=0.0005,
                 freq_mask_param=10, time_mask_param=10,):
        self.path_to_file_names = path_to_file_names
        self.path_to_files = path_to_files
        self.classes = classes
        self.sample_rate = sample_rate
        self.has_labels = has_labels
        self.augmentation_chance = augmentation_chance
        self.noise_chance = noise_chance
        self.aug_strength = aug_strength
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.mean = mean
        self.std = std
        self.mel_transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=640, hop_length=320, n_mels=40)
        self.db_transform = transforms.AmplitudeToDB()
        self.file_paths, self.labels = self._build_file_paths_and_labels()

    def _process_audio(self, audio_path, label):
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.mean(dim=0) if audio.shape[0] > 1 else audio
        audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=self.sample_rate)
        audio = self.mel_transform(audio)
        audio = self.db_transform(audio)

        # Normalize the spectrogram
        audio = (audio - self.mean) / self.std
        audio = self._apply_augmentation(audio) if random.random() < self.augmentation_chance else audio
        audio = audio.permute(1, 0)  # Swap frequency and time axes
        return audio, label

    def __getitem__(self, index):
        audio_path = self.file_paths[index % len(self.file_paths)]
        label = self.labels[index % len(self.labels)] if self.has_labels else None
        audio, label = self._process_audio(audio_path, label)
        return audio, label

    def __len__(self):
        return len(self.file_paths)

    def _build_file_paths_and_labels(self):
        file_paths, labels = [], []
        with open(self.path_to_file_names, 'r') as file:
            for line in file:
                parts = re.split(r'\t|\s', line.strip())
                if len(parts) == 2 or (len(parts) == 1 and not self.has_labels):
                    file_path = self.path_to_files + parts[0]
                    file_paths.append(file_path)
                    if self.has_labels:
                        label = self.classes[parts[1]] - 1
                        labels.append(label)
        return file_paths, labels or None

    def _apply_augmentation(self, audio):
        if random.random() < self.noise_chance:
            audio = self._add_random_noise(audio)
        else:
            audio = self._frequency_mask(audio)
            audio = self._time_mask(audio)
        return audio

    def _add_random_noise(self, audio):
        noise = torch.randn_like(audio) * self.aug_strength
        return audio + noise

    def _frequency_mask(self, audio):
        freq_mask = transforms.FrequencyMasking(self.freq_mask_param)
        return freq_mask(audio)

    def _time_mask(self, audio):
        time_mask = transforms.TimeMasking(self.time_mask_param)
        return time_mask(audio)
