import torch
from torch import nn


class Answer(nn.Module):

    def __init__(self, hidden_dim, num_classes, layers=1):
        super().__init__()
        if layers == 1:
            self.answer = nn.Sequential(
                nn.Linear(hidden_dim, num_classes),
                nn.Softmax(dim=-1)
            )
        elif layers == 2:
            self.answer = nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, num_classes),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        return self.answer(x)
