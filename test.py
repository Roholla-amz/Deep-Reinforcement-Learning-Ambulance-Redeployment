import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import *
from environment import Environment

env = Environment(m=5, k=7, calls_size=4800, ambulance_count=25, normalize=False)
num_stations = len(env.stations)
input_dim = 3 * env.m + 1 + 1 + env.k
reinforce_agent = ReinforceAgent.load('trained_policy_with_pr.pth', input_dim=input_dim, num_stations=num_stations)
ns_agent = NSAgent(m=env.m)
ls_agent = LSAgent(m=env.m)
random_agent = RandomAgent(num_stations=num_stations)
    
state = env.reset(call_start=4800)
while True:        
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action = reinforce_agent.select_best_action(state_tensor)
    next_state, reward, done = env.step(action)
    
    state = next_state

    if done:
        break

print('reinforce agent:')
print(' Average Pickup Time:', env.stats.AvePT())
print(' Relative Pickup Time:', env.stats.RelaPT())


state = env.reset(call_start=4800)
while True:        
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action = ns_agent.select_action(state_tensor)
    next_state, reward, done = env.step(action)
    
    state = next_state

    if done:
        break

print('NS agent:')
print(' Average Pickup Time:', env.stats.AvePT())
print(' Relative Pickup Time:', env.stats.RelaPT())


state = env.reset(call_start=4800)
while True:        
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action = ls_agent.select_action(state_tensor)
    next_state, reward, done = env.step(action)
    
    state = next_state

    if done:
        break

print('LS agent:')
print(' Average Pickup Time:', env.stats.AvePT())
print(' Relative Pickup Time:', env.stats.RelaPT())


state = env.reset(call_start=4800)
while True:        
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action = random_agent.select_action(state_tensor)
    next_state, reward, done = env.step(action)
    
    state = next_state

    if done:
        break

print('random agent:')
print(' Average Pickup Time:', env.stats.AvePT())
print(' Relative Pickup Time:', env.stats.RelaPT())
