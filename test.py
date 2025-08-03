import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import ReinforceAgent
from environment import Environment

env = Environment(m=5, k=7, calls_size=4800, ambulance_count=25, normalize=False)
num_stations = len(env.stations)
input_dim = 3 * env.m + 1 + 1 + env.k
agent = ReinforceAgent.load('trained_policy_with_pr.pth', input_dim=input_dim, num_stations=num_stations)
    
state = env.reset(call_start=4800)
log_probs = []
total_reward = 0
total_reward_per_priority = [0, 0, 0]

while True:        
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action = agent.select_best_action(state_tensor)
    next_state, reward, done, reward_per_priority = env.step(action)
    
    total_reward += reward
    for i in range(3):
        total_reward_per_priority[i] += reward_per_priority[i]
    state = next_state

    if done:
        break

print('Total Reward:', total_reward)
print('Total Reward per priority:', total_reward_per_priority)
