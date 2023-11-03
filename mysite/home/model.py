import torch
import torch.nn as nn


class BRNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer_block = nn.Sequential(
            nn.Embedding(input_dim, 100),
            nn.LSTM(100, 128, bidirectional=True),
            nn.Linear(256, output_dim),
            nn.Dropout(0.5)
        )
        
    def forward(self, text):
        embedded = self.layer_block[0](text)
        output, (hidden, cell) = self.layer_block[1](embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        return self.layer_block[2](hidden.squeeze(0))