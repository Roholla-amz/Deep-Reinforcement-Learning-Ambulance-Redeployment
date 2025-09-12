import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import os

from agent import ReinforceAgent
from environment import Environment

def train_with_ratio(ratio=0.7):
    print(f"Training with ratio = {ratio}")
    env = Environment(m=1, k=1, ambulance_count=15, use_map=False, verbose=False)
    
    train_size = int(len(env.calls) * ratio)
    env.call_size = train_size

    num_stations = len(env.stations)
    input_dim = 3 * env.m + 1 + 1 + env.k        
    agent = ReinforceAgent(input_dim=input_dim, num_stations=num_stations)
    reward_history = []
    relaPT_history = [] 
    ave_PT_history = [] 

    for episode in tqdm(range(40)):
        
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
        avePT = env.stats.AvePT()
        
        print(f"avePT: {[f'{x:.1f}' for x in avePT]}, relaPT: {[f'{x:.3f}' for x in relaPT]}")
        
        relaPT_history.append(relaPT)
        ave_PT_history.append(avePT)

    model_name = f'models/trained_policy_m1_k1_amb15_tr{ratio}.pth'
    agent.save(model_name)
    
    # Plot AvePT over episodes for 3 priority classes
    avePT_arr = np.array(ave_PT_history)  # shape (episodes, 3)
    episodes = np.arange(1, len(ave_PT_history) + 1)
    colors = ['green', 'orange', 'red']
    labels = ['Priority 0', 'Priority 1', 'Priority 2']

    plt.figure(figsize=(10, 5))
    for p in range(3):
        plt.plot(episodes, avePT_arr[:, p], color=colors[p], label=labels[p])
    plt.title('AvePT per Episode by Priority Class')
    plt.xlabel('Episode')
    plt.ylabel('AvePT')
    plt.legend(loc='best')
    plt.tight_layout()
    fig_path = os.path.join('graphs', f'avept_over_episodes_tr{ratio}.png')
    plt.savefig(fig_path)
    plt.close()
    print("training completed, graph saved to ", fig_path)

def train_with_differnt_mk(ratio=0.7):
    # Train for multiple (m, k) configurations and plot AvePT curves
    m_values = [1, 2, 3, 4, 5]
    k_values = [1, 2, 3, 4, 5]
    num_episodes = 40

    colors = ['green', 'orange', 'red']
    labels = ['Priority 0', 'Priority 1', 'Priority 2']
    os.makedirs('graphs', exist_ok=True)

    for m in m_values:
        for k in k_values:
            print(f"\nTraining with configuration m={m}, k={k}")
            env = Environment(m=m, k=k, ambulance_count=15, use_map=False, verbose=False)
            train_size = int(len(env.calls) * ratio)
            env.call_size = train_size

            num_stations = len(env.stations)
            input_dim = 3 * env.m + 1 + 1 + env.k
            agent = ReinforceAgent(input_dim=input_dim, num_stations=num_stations)

            ave_PT_history = []

            for episode in tqdm(range(num_episodes), desc=f'm={m}, k={k}'):
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
                avePT = env.stats.AvePT()
                ave_PT_history.append(avePT)

            # Save model per configuration
            model_name = f'models/trained_policy_m{m}_k{k}_amb15_tr{ratio}.pth'
            agent.save(model_name)

if __name__ == "__main__":
    # train_with_ratio(ratio=0.7)
    train_with_differnt_mk(ratio=0.7)