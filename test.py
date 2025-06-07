import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import ReinforceAgent
from environment import Environment

env = Environment(m=5, k=7, calls_size=4800, ambulance_count=25, normalize=True)
num_stations = len(env.stations)
input_dim = env.m + 1 + 1 + env.k
agent = ReinforceAgent.load('trained_policy.pth', input_dim=input_dim, num_stations=num_stations)
    
state = env.reset(call_start=4800)
log_probs = []
rewards = []

while True:        
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action, log_prob = agent.select_action(state_tensor)
    next_state, reward, done = env.step(action)
    
    rewards.append(reward)
    state = next_state

    if done:
        break

print('Total Reward:', sum(rewards))
