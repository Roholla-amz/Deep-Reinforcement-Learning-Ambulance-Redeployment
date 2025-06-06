import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepScoreNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.output(x)  # scalar score

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, num_stations: int):
        super().__init__()
        self.score_net = DeepScoreNetwork(input_dim)
        self.num_stations = num_stations

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: shape [num_stations, input_dim]
        returns: probs over stations [num_stations]
        """
        scores = torch.cat([self.score_net(station) for station in state], dim=0).squeeze()
        probs = F.softmax(scores, dim=0)
        return probs
