import torch
from pytorch_lightning import LightningModule, Trainer
import torchaudio
from torchaudio.models import Tacotron2

import os

from torch.utils.data import Dataset, DataLoader
import librosa

class TTSDataset(Dataset):
    def __init__(self, metadata_path, audio_dir):
        with open(r"C:\Users\swcar\Desktop\College\AiWVU\BoJack Horseman\LJSpeech-1.1\metadata.csv", 'r') as f:
            self.metadata = f.readlines()
        self.audio_dir = audio_dir
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        line = self.metadata[idx].strip().split('|')
        audio_path = f"{self.audio_dir}/{line[0]}.wav"
        text_input = line[1]
        audio_data, _ = librosa.load(audio_path)
        return text_input, audio_data






class Tacotron2TTS(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = Tacotron2(cfg)
        self.dataset = torchaudio.datasets.LJSPEECH(root=wav_file, download = False)

    def forward(self, text):
        return self.model(text)
    
    def training_step(self, batch, batch_idx):
        text, mel_spectrogram = batch
        mel_spectrogram_pred = self(text)
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, mel_spectrogram)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        text, mel_spectrogram = batch
        mel_spectrogram_pred = self(text)
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, mel_spectrogram)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        text, mel_spectrogram = batch
        mel_spectrogram_pred = self(text)
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, mel_spectrogram)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
if __name__ == '__main__':
    cfg = {
        "model": {
            "num_mels": 80,
            "hidden_channels": 128,
            "attention_dim": 128,
        },
        "trainer": {
            "max_epochs": 100,
        },
    }

    model = Tacotron2TTS(cfg)
    trainer = Trainer()
    trainer.fit(model)
