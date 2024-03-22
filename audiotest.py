import os
import torch
import torchaudio
import re
from torchaudio import transforms
import random


def add_random_noise(audio, strength):
        noise = torch.randn_like(audio) * strength
        return audio + noise

audio, sample_rate = torchaudio.load("/data/baproj/dlap/test-files-partial/test_audio_0110.wav")
if audio.shape[0] > 1: 
    audio = audio.mean(dim=0)
audio = torchaudio.functional.resample(audio, orig_freq=44100, new_freq=44100)


audio_noise = add_random_noise(audio, 0.0005)

# Erstellen des Ordners audiotest, falls nicht vorhanden
if not os.path.exists('audiotest'):
    os.makedirs('audiotest')

# Speichern der Audiodateien
torchaudio.save('audiotest/clean_audio.wav', audio.unsqueeze(0), 44100)
torchaudio.save('audiotest/noisy_audio.wav', audio_noise.unsqueeze(0), 44100)

