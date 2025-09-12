import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import *
from environment import Environment



def compare_models_over_ambulance_count(ratio=0.7):
    env = Environment(m=1, k=1, ambulance_count=15, use_map=False, verbose=False)
    call_size = int(len(env.calls) * (1-ratio))

    num_stations = len(env.stations)
    input_dim = 3 * env.m + 1 + 1 + env.k
    reinforce_agent = ReinforceAgent.load('trained_policy_m1_k1_amb15_cs4800.pth', input_dim=input_dim, num_stations=num_stations)
    ls_agent = LSAgent(m=env.m)
    ns_agent = NSAgent(m=env.m)
    random_agent = RandomAgent(num_stations=num_stations)

    # Experiment: vary ambulance_count from 5 to 35 and compare Reinforce vs LS
    ambulance_counts = list(range(5, 36))

    reinforce_avept = []
    reinforce_relapt = []
    reinforce_90th = []

    ls_avept = []
    ls_relapt = []
    ls_90th = []

    ns_avept = []
    ns_relapt = []
    ns_90th = []

    rand_avept = []
    rand_relapt = []
    rand_90th = []

    for ac in tqdm(ambulance_counts, desc='Evaluating over ambulance_count'):
        # Run Reinforce agent
        state = env.reset(call_start=call_size+1, call_size=call_size)
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = reinforce_agent.select_best_action(state_tensor)
            next_state, reward, done = env.step(action)
            state = next_state
            if done:
                break
        reinforce_avept.append(env.stats.AvePT()[2])
        reinforce_relapt.append(env.stats.RelaPT()[2])
        reinforce_90th.append(env.stats.P90()[2])
        # Run LS agent
        state = env.reset(call_start=call_size+1, call_size=call_size)
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = ls_agent.select_action(state_tensor)
            next_state, reward, done = env.step(action)
            state = next_state
            if done:
                break
        ls_avept.append(env.stats.AvePT()[2])
        ls_relapt.append(env.stats.RelaPT()[2])
        ls_90th.append(env.stats.P90()[2])
        # Run NS agent
        state = env.reset(call_start=call_size+1, call_size=call_size)
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = ns_agent.select_action(state_tensor)
            next_state, reward, done = env.step(action)
            state = next_state
            if done:
                break
        ns_avept.append(env.stats.AvePT()[2])
        ns_relapt.append(env.stats.RelaPT()[2])
        ns_90th.append(env.stats.P90()[2])
        # Run Random agent
        state = env.reset(call_start=call_size+1, call_size=call_size)
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = random_agent.select_action(state_tensor)
            next_state, reward, done = env.step(action)
            state = next_state
            if done:
                break
        rand_avept.append(env.stats.AvePT()[2])
        rand_relapt.append(env.stats.RelaPT()[2])
        rand_90th.append(env.stats.P90()[2])
    # Plot Average Pickup Time vs ambulance_count
    plt.figure(figsize=(8, 5))
    plt.plot(ambulance_counts, reinforce_avept, label='Reinforce - AvePT')
    plt.plot(ambulance_counts, ls_avept, label='LS - AvePT')
    plt.plot(ambulance_counts, ns_avept, label='NS - AvePT')
    plt.plot(ambulance_counts, rand_avept, label='Random - AvePT')
    plt.xlabel('Ambulance Count')
    plt.ylabel('Average Pickup Time')
    plt.title('Average Pickup Time vs Ambulance Count')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/avept_vs_ambulance_count.png', dpi=150)
    plt.close()

    # Plot Relative Pickup Time vs ambulance_count
    plt.figure(figsize=(8, 5))
    plt.plot(ambulance_counts, reinforce_relapt, label='Reinforce - RelaPT')
    plt.plot(ambulance_counts, ls_relapt, label='LS - RelaPT')
    plt.plot(ambulance_counts, ns_relapt, label='NS - RelaPT')
    plt.plot(ambulance_counts, rand_relapt, label='Random - RelaPT')
    plt.xlabel('Ambulance Count')
    plt.ylabel('Relative Pickup Time')
    plt.title('Relative Pickup Time vs Ambulance Count')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/relapt_vs_ambulance_count.png', dpi=150)
    plt.close()

    # Plot 90-th percentile Pickup Time vs ambulance_count
    plt.figure(figsize=(8, 5))
    plt.plot(ambulance_counts, reinforce_90th, label='Reinforce - 90th')
    plt.plot(ambulance_counts, ls_90th, label='LS - 90th')
    plt.plot(ambulance_counts, ns_90th, label='NS - 90th')
    plt.plot(ambulance_counts, rand_90th, label='Random - 90th')
    plt.xlabel('Ambulance Count')
    plt.ylabel('90-th percentile Pickup Time')
    plt.title('90-th percentile Pickup Time vs Ambulance Count')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/90th_vs_ambulance_count.png', dpi=150)
    plt.close()

def compare_model_over_mk():
    env = Environment(m=1, k=1, ambulance_count=15, use_map=False, verbose=False)
    call_size = int(len(env.calls) * (1-ratio))

    num_stations = len(env.stations)
    input_dim = 3 * env.m + 1 + 1 + env.k
    reinforce_agent = ReinforceAgent.load('trained_policy_m1_k1_amb15_cs4800.pth', input_dim=input_dim, num_stations=num_stations)
    # ls_agent = LSAgent(m=env.m)
    # ns_agent = NSAgent(m=env.m)
    # random_agent = RandomAgent(num_stations=num_stations)

    for i in range(2):
        for x in range(1, 6):
            m = 1
            k = 1
            if i == 0:
                m = x
            else:
                k = x
            state = env.reset(call_start=call_size+1, call_size=call_size, m=m, k=k)
            input_dim = 3 * env.m + 1 + 1 + env.k
            reinforce_agent = ReinforceAgent.load(f'./models/trained_policy_m{m}_k{k}.pth', input_dim=input_dim, num_stations=num_stations)

            while True:        
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = reinforce_agent.select_best_action(state_tensor)
                next_state, reward, done = env.step(action)
                
                state = next_state

                if done:
                    break

            print(f'm={m} k={k}:')
            print(' Average Pickup Time', ' Relative Pickup Time:', ' 90-th percentile Pickup Time:')
            print(' ', env.stats.AvePT(), env.stats.RelaPT(), env.stats.P90())

def test_with_noise():
    # Test each model with the same noise value it was trained on
    noise_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    env = Environment(m=1, k=1, calls_size=4800, ambulance_count=15, use_map=False, verbose=False)
    num_stations = len(env.stations)
    input_dim = 3 * env.m + 1 + 1 + env.k
    
    for noise in noise_values:
        print(f"\nTesting model trained with noise = {noise:.1f}")
        
        # Load the model trained on this specific noise value
        model_name = f'models/trained_policy_m1_k1_amb15_cs3500_noise{noise:.1f}.pth'
        
        reinforce_agent = ReinforceAgent.load(model_name, input_dim=input_dim, num_stations=num_stations)
        
        # Test with the same noise value the model was trained on
        state = env.reset(call_start=4800, noise=noise)
        while True:        
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = reinforce_agent.select_best_action(state_tensor)
            next_state, reward, done = env.step(action)
            
            state = next_state

            if done:
                break

        print(f'Model trained on noise={noise:.1f}, tested on noise={noise:.1f}:')
        print(' Average Pickup Time:', env.stats.AvePT())
        print(' Relative Pickup Time:', env.stats.RelaPT())
        print(' 90-th percentile Pickup Time:', env.stats.P90())
            
def compare_models(ratio=0.7):
    env = Environment(m=1, k=1, ambulance_count=15, use_map=False, verbose=False)
    call_size = int(len(env.calls) * (1-ratio))

    num_stations = len(env.stations)
    input_dim = 3 * env.m + 1 + 1 + env.k
    reinforce_agent = ReinforceAgent.load('trained_policy_m1_k1_amb15_cs4800.pth', input_dim=input_dim, num_stations=num_stations)
    ns_agent = NSAgent(m=env.m)
    ls_agent = LSAgent(m=env.m)
    random_agent = RandomAgent(num_stations=num_stations)
    
    
    state = env.reset(call_start=call_size+1, call_size=call_size)
    while True:        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = reinforce_agent.select_best_action(state_tensor)
        next_state, reward, done = env.step(action)
        
        state = next_state

        if done:
            break

    print('Reinforce agent')
    print(' Average Pickup Time:', env.stats.AvePT())
    print(' Relative Pickup Time:', env.stats.RelaPT())
    print(' 90-th percentile Pickup Time:', env.stats.P90())
    
    state = env.reset(call_start=call_size+1, call_size=call_size)
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
    print(' 90-th percentile Pickup Time:', env.stats.P90())

    
    state = env.reset(call_start=call_size+1, call_size=call_size)
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
    print(' 90-th percentile Pickup Time:', env.stats.P90())

    state = env.reset(call_start=call_size+1, call_size=call_size)
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
    print(' 90-th percentile Pickup Time:', env.stats.P90())

if __name__ == "__main__":
    # compare_model_over_mk()
    # compare_models()
    test_with_noise()