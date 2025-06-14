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

# NN with 3 layers
class L3_QNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(L3_QNet, self).__init__()
        self.layers = 3
        self.input_layer = nn.Linear(state_dim, 48)
        self.hidden_layer = nn.Linear(48, 48)
        self.output_layer = nn.Linear(48, action_dim)

    def forward(self, state):
        out1 = torch.relu(self.input_layer(state))
        out2 = torch.relu(self.hidden_layer(out1))
        return self.output_layer(out2)


# NN with 4 layers
class L4_QNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(L4_QNet, self).__init__()
        self.layers = 4
        self.input_layer = nn.Linear(state_dim, 48)
        self.hidden_layer_2 = nn.Linear(48, 48)
        self.hidden_layer_1 = nn.Linear(48, 48)
        self.output_layer = nn.Linear(48, action_dim)

    def forward(self, state):
        out1 = torch.relu(self.input_layer(state))
        out2 = torch.relu(self.hidden_layer_1(out1))
        out3 = torch.relu(self.hidden_layer_2(out2))
        return self.output_layer(out3)
    
# NN with 5 layers
class L5_QNet(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(L5_QNet, self).__init__()
        self.layers = 5
        self.input_layer = nn.Linear(state_dim, 48)
        self.hidden_layer_1 = nn.Linear(48, 48)
        self.hidden_layer_2 = nn.Linear(48, 48)
        self.hidden_layer_3 = nn.Linear(48, 48)
        self.output_layer = nn.Linear(48, action_dim)

    def forward(self, state):
        out1 = torch.relu(self.input_layer(state))
        out2 = torch.relu(self.hidden_layer_1(out1))
        out3 = torch.relu(self.hidden_layer_2(out2))
        out4 = torch.relu(self.hidden_layer_3(out3))
        return self.output_layer(out4)


# Replay buffer used by NNs (used as a queue)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def pick(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def sizeof(self):
        return len(self.buffer)

# Agent for NN
class Agent:
    def __init__(self, layers, state_dim, action_dim, gamma, epsilon, epsilon_decay, epsilon_min, lr):
        
        self.state_dim = state_dim      # Dimensions of state
        self.action_dim = action_dim    # Possible actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.layers = layers
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # matching NN for given number of layers
        match(layers):
            case 3:    
                self.q_network = L3_QNet(state_dim, action_dim).to(self.device)
                self.target_network = L3_QNet(state_dim, action_dim).to(self.device)

            case 4:
                self.q_network = L4_QNet(state_dim, action_dim).to(self.device)
                self.target_network = L4_QNet(state_dim, action_dim).to(self.device)
                
            case 5:
                self.q_network = L5_QNet(state_dim, action_dim).to(self.device)
                self.target_network = L5_QNet(state_dim, action_dim).to(self.device)

        # set device to elaborate information
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(10000)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # random selection action by a given state
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                # conversion in tensors and in "one_hot"s
                state_tensor = torch.tensor([state], dtype=torch.long, device=self.device)
                state_tensor = F.one_hot(state_tensor, num_classes=self.state_dim).float()
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

   # training of network by using the agent
    def replay(self, batch_size=64):
        if self.replay_buffer.sizeof() < batch_size:
            return None
        
        batch = self.replay_buffer.pick(batch_size)

        states = [item[0] for item in batch]
        actions = [item[1] for item in batch]
        rewards = [item[2] for item in batch]
        next_states = [item[3] for item in batch]
        dones = [item[4] for item in batch]

        # transforming current and future state in tensors + correcting size
        states = torch.tensor(states, dtype=torch.long, device=self.device)
        states = F.one_hot(states, num_classes=self.state_dim).float()
        next_states = torch.tensor(next_states, dtype=torch.long, device=self.device)
        next_states = F.one_hot(next_states, num_classes=self.state_dim).float()

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)

        # actual q-values and q-targets
        q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)

        # q-values related to corresponding done actions
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        target = rewards + (self.gamma * next_q_value * (1 - dones))

        loss = F.mse_loss(q_value, target)

        # network optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def save_train(self):
        torch.save(self.target_network.state_dict(), f'./trains/{self.layers}L.pth')
    
    def load_train(self, filename):
        filepath = str("./trains/" + filename)
        print("Searching for " + filepath + " ...")
        
        print(os.path.isfile(filepath))
        
        if(os.path.isfile(filepath)):
                self.q_network.load_state_dict(torch.load(filepath, weights_only=True))
                self.target_network.load_state_dict(torch.load(filepath, weights_only=True))
                
    # plotting rewards using mobile window of 10 episodes
    def plot_rewards_smoothed(self, rewards, layers):
        window = 10
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.3, label="Raw Reward")
        plt.plot(range(window - 1, len(rewards)), smoothed_rewards, label=f"Smoothed Rewards (window={window})", color='green')

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.suptitle(f"Rewards Curve for NN with {self.layers} layers\n")
        plt.title(f"γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%(self.epsilon)}, ε_dec = {'%.4f'%(self.epsilon_decay)}, ε_min = {'%.4f'%(self.epsilon_min)}, lr = {'%.4f'%(self.lr)}")

        plt.savefig(f"{layers}L_rewards.jpg")
        plt.clf()

    # plotting losses using mobile window of 10 episodes
    def plot_losses(self, losses, layers):
        window = 10
        smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
        
        plt.figure(figsize=(10, 5))
        plt.plot(losses, alpha=0.3, label="Raw Loss")
        plt.plot(range(window - 1, len(losses)), smoothed_losses, label=f"Smoothed Loss (window={window})", color='red')

        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.suptitle(f"Loss Curve for NN with {self.layers} layers\n")
        plt.title(f"γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%(self.epsilon)}, ε_dec = {'%.4f'%(self.epsilon_decay)}, ε_min = {'%.4f'%(self.epsilon_min)}, lr = {'%.4f'%(self.lr)}")

        plt.savefig(f"{layers}L_loss.jpg")
        plt.clf()
        
    # plotting accuracy
    def plot_accuracy(self, successes):
        size = len(successes)
        tmp = 0
        accuracy = []
        window = 10
        
        for i in range(1, size + 1):
            tmp += successes[i-1]
            print(f"Accuracy: {tmp/i}")
            accuracy.append(tmp/i)
            
        smoothed_accuracy = np.convolve(accuracy, np.ones(window)/window, mode='valid')
        
        plt.plot(accuracy, alpha=0.3, label="Raw Accuracy")
        plt.plot(range(window - 1, len(accuracy)), smoothed_accuracy, label=f"Smoothed Loss (window={window})", color='blue')
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.suptitle(f"Accuracy Curve for NN with {self.layers} layers\n")
        plt.title(f"γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%(self.epsilon)}, ε_dec = {'%.4f'%(self.epsilon_decay)}, ε_min = {'%.4f'%(self.epsilon_min)}, lr = {'%.4f'%(self.lr)}")
        
        plt.savefig(f"{self.layers}L_accuracy.jpg")
        plt.clf()

# training function
def train(episodes, gamma, epsilon, epsilon_decay, epsilon_min, lr):
    env = gym.make("Taxi-v3")
    
    L3_agent = Agent(layers = 3,
                     state_dim=env.observation_space.n,
                     action_dim=env.action_space.n,
                     gamma = gamma, 
                     epsilon = epsilon, 
                     epsilon_decay= epsilon_decay,
                     epsilon_min = epsilon_min, 
                     lr = lr)
    
    L4_agent = Agent(layers = 4,
                     state_dim=env.observation_space.n,
                     action_dim=env.action_space.n,
                     gamma = gamma, 
                     epsilon = epsilon, 
                     epsilon_decay= epsilon_decay,
                     epsilon_min = epsilon_min, 
                     lr = lr)
    
    L5_agent = Agent(layers = 5,
                     state_dim=env.observation_space.n,
                     action_dim=env.action_space.n,
                     gamma = gamma, 
                     epsilon = epsilon, 
                     epsilon_decay= epsilon_decay,
                     epsilon_min = epsilon_min, 
                     lr = lr)

    agents = [L3_agent, L4_agent, L5_agent]
        
    for a in agents:
        rewards = []
        losses_per_episode = [] 
        successes = []

        # a.load_train(str(a.layers) + "L.pth")

        for e in range(episodes):
            state, _ = env.reset() # for obtaining (observation, info) tuple
            done = False
            total_reward = 0
            episode_losses = []  # loss per episodes stored
            success = 0
            
            while not done:
                action = a.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                if(terminated):
                    success = 1

                done = terminated or truncated

                # store states, update state and total reward
                a.replay_buffer.push(state, action, reward, next_state, done)
                loss_val = a.replay(batch_size=64)
                if loss_val is not None:
                    episode_losses.append(loss_val)
                state = next_state
                total_reward += reward

            # each 10 episodes update target network
            if e % 10 == 0:
                a.update_target_network()
            
            # avg loss per episode
            if len(episode_losses) > 0:
                avg_loss = np.mean(episode_losses)
                losses_per_episode.append(avg_loss)
            else:
                losses_per_episode.append(0)

            # epsilon update
            if a.epsilon > a.epsilon_min:
                a.epsilon *= a.epsilon_decay
            
            successes.append(success)            
            rewards.append(total_reward)
            print(f"Agent {a.layers}L - Episode {e}, Total Reward: {total_reward}, ε: {a.epsilon:.4f}, Avg Loss: {losses_per_episode[-1]:.6f}")
        
        # rewards and losses plotting
        a.plot_rewards_smoothed(rewards, a.layers)
        a.plot_losses(losses_per_episode, a.layers)
        a.plot_accuracy(successes)
        
        a.save_train()
        
        successes = None
        successes = []

if __name__ == "__main__":

    env = gym.make("Taxi-v3")    
    state_dim=env.observation_space.n,
    action_dim=env.action_space.n,
    
    # hyperparameters 1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    lr = 0.001
    
    # hyperparameters 2
    # gamma = 0.99
    # epsilon = 1.0
    # epsilon_decay = 0.998
    # epsilon_min = 0.1
    # lr = 0.0005

    episodes = 8000
    train(episodes, gamma, epsilon, epsilon_decay, epsilon_min, lr)
