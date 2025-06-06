import torch
import torch.optim as optim
from typing import List, Tuple

from network import PolicyNetwork


class ReinforceAgent:
    def __init__(self, input_dim: int, num_stations: int, lr: float = 0.005):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(input_dim, num_stations)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Given a state, sample an action and return its log-probability.
        """
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, log_probs: List[torch.Tensor], rewards: List[float]):
        """
        Update policy using REINFORCE with baseline.
        """
        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Use average return as baseline
        baseline = returns.mean()
        loss = -torch.sum(torch.stack(log_probs) * (returns - baseline))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
