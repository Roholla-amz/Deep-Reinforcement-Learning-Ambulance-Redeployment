import torch
import torch.optim as optim
from typing import List, Tuple
import random
from network import PolicyNetwork


class ReinforceAgent:
    def __init__(self, input_dim: int, num_stations: int, lr: float = 0.005):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(input_dim, num_stations)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99
        self.entropy_coef = 0.001

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Given a state, sample an action and return its log-probability.
        """
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
    
    def select_best_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Given a state, select the best action
        """
        probs = self.policy.forward(state)
        highest_prob, action = torch.max(probs, dim=0)
        return action.item()

    def update(self, log_probs: List[torch.Tensor], rewards: List[float], entropies: List[torch.Tensor]):
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
        
        adv = returns - returns.mean()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        logp = torch.stack(log_probs)
        ent   = torch.stack(entropies)
        entropy_bonus = ent.sum() * self.entropy_coef
        
        actor_loss = - (logp * adv).sum()
        loss = actor_loss - entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path: str):
        """
        Saves just the policy network's state dict to disk.
        """
        torch.save(self.policy.state_dict(), path)

    @classmethod
    def load(cls, path: str, input_dim: int, num_stations: int):
        """
        Constructs a new agent, loads weights, and sets to eval mode.
        """
        agent = cls(input_dim=input_dim, num_stations=num_stations)
        state_dict = torch.load(path)
        agent.policy.load_state_dict(state_dict)
        agent.policy.eval()
        
        return agent
    
class RandomAgent:
    def __init__(self, num_stations: int):
        self.num_stations = num_stations

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Select a random action from the available stations.
        """
        action = random.randint(0, self.num_stations - 1)
        return action

class NSAgent:
    def __init__(self, m: int):
        self.m = m

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Select the nearest station based on the state.
        """
        action = [0]
        tt = int(1e9)
        for i, s in enumerate(state):
            if s[3*self.m + 1] < tt:
                tt = s[self.m + 1]
                action = [i]
            elif s[3*self.m + 1] == tt:
                action.append(i)
        return random.choice(action)

class LSAgent:
    def __init__(self, m: int):
        self.m = m

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Select the station with the least number of ambulances assigned.
        """
        action = [0]
        min_ambulances = int(1e9)
        for i, s in enumerate(state):
            if s[3*self.m] < min_ambulances:
                min_ambulances = s[self.m + 1]
                action = [i]
            elif s[3*self.m] == min_ambulances:
                action.append(i)
        return random.choice(action)



