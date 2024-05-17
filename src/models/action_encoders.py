import math
from functools import partial

import torch
import torch.nn as nn


class ActionEncoderDiscrete(nn.Module):
    def __init__(self, num_actions, embed_dim, hidden_dim):
        super(ActionEncoderDiscrete, self).__init__()
        self.embedding = nn.Embedding(num_actions, embed_dim)
        self.linear = nn.Linear(embed_dim, hidden_dim)

    def forward(self, actions):
        embedded_actions = self.embedding(actions)
        encoded_actions = self.linear(embedded_actions)
        return encoded_actions


class ActionEncoderContinuous(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ActionEncoderContinuous, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, actions):
        encoded_actions = self.mlp(actions)
        return encoded_actions
