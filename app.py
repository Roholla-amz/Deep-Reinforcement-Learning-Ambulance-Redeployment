import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import ReinforceAgent
from environment import Environment

env = Environment(m=5, k=7, calls_size=3000, ambulance_count=35, normalize=False)
num_stations = len(env.stations)
input_dim = 3 * env.m + 1 + 1 + env.k
agent = ReinforceAgent(input_dim=input_dim, num_stations=num_stations)

reward_history = []
relaPT_history = []

state_history = []
for episode in tqdm(range(1, 30 + 1)):
    
    state = env.reset()
    log_probs = []
    rewards = []
    entropies = []

    while True:        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action, log_prob, entropy = agent.select_action(state_tensor)
        next_state, reward, done = env.step(action)
        
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)
        state = next_state

        if done:
            break

    agent.update(log_probs, rewards, entropies)
    
    reward_history.append(sum(rewards)) 
    relaPT = env.stats.RelaPT()
    print(relaPT)
    relaPT_history.append(relaPT)

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed = moving_average(reward_history, window_size=5)

plt.figure(figsize=(10, 5))
plt.plot(reward_history, alpha=0.3, label='Raw Reward')
plt.plot(smoothed, label='Smoothed (moving avg)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward over Episodes (Smoothed)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot AvePT for each priority ---
plt.figure(figsize=(10, 5))
plt.plot(relaPT_history, label='RelaPT')
plt.xlabel('Episode')
plt.ylabel('RelaPT (percent)')
plt.title('RelaPT per Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

model_path = 'trained_policy_with_pr4.pth'
agent.save(model_path)
