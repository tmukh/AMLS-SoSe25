import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class STFT_CNN_RNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = 128
        self.hop_length = 16

        self.conv1 = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2)
        )


        # Dummy forward pass to compute feature size
        dummy_signal = torch.randn(1, 3000)  # Simulate ~10-second ECG
        dummy_stft = torch.stft(dummy_signal, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        dummy_x = torch.abs(dummy_stft).log1p().unsqueeze(1)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.conv2(dummy_x)
        B, C, Freq, Time = dummy_x.size()
        feature_size = C * Freq

        self.rnn = nn.GRU(input_size=feature_size, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 4)

    def forward(self, x, lengths):
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        x = torch.abs(stft).log1p().unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)

        B, C, Freq, Time = x.size()
        x = x.view(B, C * Freq, Time).permute(0, 2, 1)

        lengths = torch.div(lengths, self.hop_length, rounding_mode='floor')
        lengths = lengths // 4
        lengths = torch.clamp(lengths, min=1, max=Time)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        output = self.fc(hidden[-1])
        return output
