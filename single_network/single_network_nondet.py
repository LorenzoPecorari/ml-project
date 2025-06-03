import gymnasium as gym
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import os

class L3_QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(L3_QNet, self).__init__()
        self.layers=3
        self.input_layer=nn.Linear(state_dim, 64)
        self.hidden_layer=nn.Linear(64, 64)
        self.output_layer=nn.Linear(64, action_dim)

    def forward(self, state):
        out1=torch.relu(self.input_layer(state))
        out2=torch.relu(self.hidden_layer(out1))
        return self.output_layer(out2)

class L4_QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(L4_QNet, self).__init__()
        self.layers=4
        self.input_layer=nn.Linear(state_dim, 64)
        self.hidden_layer_1=nn.Linear(64, 64)
        self.hidden_layer_2=nn.Linear(64, 64)
        self.output_layer=nn.Linear(64, action_dim)

    def forward(self, state):
        out1=torch.relu(self.input_layer(state))
        out2=torch.relu(self.hidden_layer_1(out1))
        out3=torch.relu(self.hidden_layer_2(out2))
        return self.output_layer(out3)

class L5_QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(L5_QNet, self).__init__()
        self.layers=5
        self.input_layer=nn.Linear(state_dim, 64)
        self.hidden_layer_1=nn.Linear(64, 64)
        self.hidden_layer_2=nn.Linear(64, 64)
        self.hidden_layer_3=nn.Linear(64, 64)
        self.output_layer=nn.Linear(64, action_dim)

    def forward(self, state):
        out1=torch.relu(self.input_layer(state))
        out2=torch.relu(self.hidden_layer_1(out1))
        out3=torch.relu(self.hidden_layer_2(out2))
        out4=torch.relu(self.hidden_layer_3(out3))
        return self.output_layer(out4)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer=deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def pick(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sizeof(self):
        return len(self.buffer)

class Agent:
    def __init__(self, layers, state_dim, action_dim, gamma, epsilon, epsilon_decay, epsilon_min, lr):
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.gamma=gamma
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min
        self.lr=lr
        self.layers=layers
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accuracy=[]
        self.losses=[]
        self.rewards=[]

        match(layers):
            case 3:
                self.q_network=L3_QNet(state_dim, action_dim).to(self.device)
            case 4:
                self.q_network=L4_QNet(state_dim, action_dim).to(self.device)
            case 5:
                self.q_network=L5_QNet(state_dim, action_dim).to(self.device)

        self.q_network.to(self.device)
        self.optimizer=optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer=ReplayBuffer(100000)

    def select_action(self, state):
        if np.random.rand()<=self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor=torch.tensor([state], dtype=torch.long, device=self.device)
                state_tensor=F.one_hot(state_tensor, num_classes=self.state_dim).float()
                q_values=self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def replay(self, batch_size):
        if self.replay_buffer.sizeof() < batch_size:
            return None

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        probs = []

        batch=self.replay_buffer.pick(batch_size)
        for item in batch:
            states.append(item[0])
            actions.append(item[1])
            rewards.append(item[2])
            next_states.append(item[3])
            dones.append(item[4])
        
        targets = []
        for i in range(len(states)):
            transitions = env.unwrapped.P[states[i]][actions[i]]
            target_i = 0.0

            for elem in transitions:
                prob = elem[0]
                ns = elem[1]
                reward = elem[2]

                ns_tensor = torch.tensor([ns], dtype=torch.long, device=self.device)
                ns_tensor = F.one_hot(ns_tensor, num_classes=self.state_dim).float()

                next_q = self.q_network(ns_tensor).max().item()
                target_i += prob * (reward + self.gamma * next_q)

            targets.append(target_i)

        targets = torch.tensor(targets, dtype=torch.float, device=self.device).detach()

        # next_states_tmp = []

        # for i in range(0, len(states)):
        #     next_states_tmp.append(env.unwrapped.P[states[i]][actions[i]])

        # for elem in next_states_tmp:
        #     probs.append(elem[0])

        
        probs=torch.tensor(probs, dtype=torch.float, device=self.device)
            
        states=torch.tensor(states, dtype=torch.long, device=self.device)
        states= F.one_hot(states, num_classes=self.state_dim).float()
        # next_states=torch.tensor(next_states,  dtype=torch.long, device=self.device)

        # # for elem in next_states:
        # #     print(elem.item())
        
        # next_states=F.one_hot(next_states, num_classes=self.state_dim).float()

        actions=torch.tensor(actions, dtype=torch.long, device=self.device)
        # rewards=torch.tensor(rewards, dtype=torch.float, device=self.device)
        # dones=torch.tensor(dones, dtype=torch.float, device=self.device)

        q_values=self.q_network(states)
        # next_q_values=self.q_network(next_states)



        q_value=q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # next_q_value=next_q_values.max(1)[0]
        # target=(probs * (rewards+(self.gamma * next_q_value))).detach()
        loss=F.mse_loss(q_value, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_train(self):
        torch.save(self.q_network.state_dict(), f'./trains/{self.layers}L.pth')

    def load_train(self, filename):
        filepath = str("./trains/" + filename)
        print("Searching for " + filepath + " ...")
        
        print(os.path.isfile(filepath))
        
        if(os.path.isfile(filepath)):
                self.q_network.load_state_dict(torch.load(filepath, weights_only=True))

    # plotting rewards using mobile window of 10 episodes
    def plot_rewards_smoothed(self):
        window = 10
        smoothed_rewards = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards, alpha=0.3, label="Raw Reward")
        plt.plot(range(window - 1, len(self.rewards)), smoothed_rewards, label=f"Smoothed Rewards (window={window})", color='green')

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.suptitle(f"Rewards Curve for NN with {self.layers} layers\n")
        plt.title(f"γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%(self.epsilon)}, ε_dec = {'%.4f'%(self.epsilon_decay)}, ε_min = {'%.4f'%(self.epsilon_min)}, lr = {'%.4f'%(self.lr)}")

        plt.savefig(f"{self.layers}L_rewards.pdf")
        plt.clf()

    # plotting losses using mobile window of 10 episodes
    def plot_losses(self):
        window = 10
        smoothed_losses = np.convolve(self.losses, np.ones(window)/window, mode='valid')
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, alpha=0.3, label="Raw Loss")
        plt.plot(range(window - 1, len(self.losses)), smoothed_losses, label=f"Smoothed Loss (window={window})", color='red')

        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.suptitle(f"Loss Curve for NN with {self.layers} layers\n")
        plt.title(f"γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%(self.epsilon)}, ε_dec = {'%.4f'%(self.epsilon_decay)}, ε_min = {'%.4f'%(self.epsilon_min)}, lr = {'%.4f'%(self.lr)}")

        plt.savefig(f"{self.layers}L_loss.pdf")
        plt.clf()
        
    # plotting accuracy
    def plot_accuracy(self):
        window = 10
        smoothed_accuracy = np.convolve(self.accuracy, np.ones(window)/window, mode='valid')
        
        plt.plot(self.accuracy, alpha=0.3, label="Raw Accuracy")
        plt.plot(range(window - 1, len(self.accuracy)), smoothed_accuracy, label=f"Smoothed Loss (window={window})", color='blue')
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.suptitle(f"Accuracy Curve for NN with {self.layers} layers\n")
        plt.title(f"γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%(self.epsilon)}, ε_dec = {'%.4f'%(self.epsilon_decay)}, ε_min = {'%.4f'%(self.epsilon_min)}, lr = {'%.4f'%(self.lr)}")
        
        plt.savefig(f"{self.layers}L_accuracy.pdf")
        plt.clf()
        

def plot_agents(agents):
    window = 100
    plt.figure(figsize=(10, 5))
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.grid()
    for agent in agents:
        plt.plot(range(window - 1, len(agent.rewards)), np.convolve(agent.rewards, np.ones(window)/window, mode='valid'), label=f"Agent {agent.layers} layers")
    plt.legend()
    plt.title("Rewards comparison")
    plt.savefig("rewards.pdf")
    
    plt.figure(figsize=(10, 5))
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.grid()
    plt.ylim(-400, 20)
    for agent in agents:
        plt.plot(range(window - 1, len(agent.rewards)), np.convolve(agent.rewards, np.ones(window)/window, mode='valid'), label=f"Agent {agent.layers} layers")
    plt.legend()
    plt.title("Rewards zoomed comparison")
    plt.savefig("rewards_zoomed.pdf")
    
    plt.figure(figsize=(10, 5))
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid()
    for agent in agents:
        plt.plot(range(window - 1, len(agent.losses)), np.convolve(agent.losses, np.ones(window)/window, mode='valid'), label=f"Agent {agent.layers} layers")
    plt.legend()
    plt.title("Losses comparison")
    plt.savefig("losses.pdf")
    
    plt.figure(figsize=(10, 5))
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.grid()
    for agent in agents:
        plt.plot(range(window - 1, len(agent.accuracy)), np.convolve(agent.accuracy, np.ones(window)/window, mode='valid'), label=f"Agent {agent.layers} layers")
    plt.legend()
    plt.title("Accuracies comparison")
    plt.savefig("accuracies.pdf")        

def train(epsiodes, gamma, epsilon, epsilon_decay, epsilon_min, lr):
    env=gym.make("Taxi-v3")
    
    L3_agent= Agent(layers=3,
                    state_dim=env.observation_space.n,
                    action_dim=env.action_space.n,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    epsilon_min=epsilon_min,
                    lr=lr)
    
    L4_agent= Agent(layers=4,
                    state_dim=env.observation_space.n,
                    action_dim=env.action_space.n,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    epsilon_min=epsilon_min,
                    lr=lr)

    L5_agent= Agent(layers=5,
                    state_dim=env.observation_space.n,
                    action_dim=env.action_space.n,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    epsilon_min=epsilon_min,
                    lr=lr)

    agents=[L3_agent, L4_agent, L5_agent]

    for a in agents:
        success=0
        for e in range(epsiodes):
            state, _ =env.reset()
            done=False
            total_reward=0
            episode_losses=[]
            

            while not done:
                action=a.select_action(state)
                next_state, reward, terminated, truncated, _ =env.step(action)

                if(terminated):
                    success+=1

                done= terminated or truncated
                a.replay_buffer.push(state, action, reward, next_state, done)
                loss_val=a.replay(64)
                if loss_val is not None:
                    episode_losses.append(loss_val)
                state=next_state
                total_reward+=reward

            if a.epsilon>a.epsilon_min:
                a.epsilon*=a.epsilon_decay

            if len(episode_losses)>0:
                avg_loss=np.mean(episode_losses)
                a.losses.append(avg_loss)
            else:
                a.losses.append(0)

            a.accuracy.append(success/(e+1))
            a.rewards.append(total_reward)
            print(f"Agent {a.layers}L - Episode {e}, Total Reward: {total_reward}, ε: {a.epsilon:.4f}, Avg Loss: {a.losses[-1]:.6f}")
        
        # a.plot_rewards_smoothed()
        # a.plot_losses()
        # a.plot_accuracy()
        
        a.save_train()

    plot_agents(agents)

if __name__=="__main__":
    env=gym.make("Taxi-v3")
    state_dim=env.observation_space.n,
    action_dim=env.action_space.n,

    gamma=0.99
    epsilon=1.0
    epslion_decay=0.995
    epsilon_min=0.1
    lr=0.001
    episodes=2500
    train(episodes, gamma, epsilon, epslion_decay, epsilon_min, lr)
