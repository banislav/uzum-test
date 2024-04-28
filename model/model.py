import torch
import torch.nn as nn

class ModelV1(nn.Module):
    def __init__(self, in_shape=3, out_shape=5, hidden_units=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)