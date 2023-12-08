import torch
from torch import nn


class Answer(nn.Module):

    def __init__(self, hidden_dim, num_classes, answer_layers=1, *args, **kwargs):
        super().__init__()
        if answer_layers <= 0:
            raise Exception('The number of answer layer can not be less than 1.')
        if answer_layers == 1:
            self.answer = nn.Sequential(
                nn.Linear(hidden_dim, num_classes),
                nn.Softmax(dim=-1)
            )
        elif answer_layers == 2:
            self.answer = nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, num_classes),
                nn.Softmax(dim=-1)
            )
        else:
            self.answer = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU())
            for _ in range(answer_layers - 2):
                self.answer.append(nn.Linear(16, 16))
                self.answer.append(nn.ReLU())

    def forward(self, x):
        return self.answer(x)
