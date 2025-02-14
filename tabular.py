import gymnasium as gym
import random

import numpy as np
import matplotlib.pyplot as plt

class QTable:
    def __init__(self, alpha, states, actions):
        self.table = np.zeros((states, actions))
        self.alpha = alpha

    def update(self, state, action, reward, next_state, gamma):
        self.table[int(state)][action] = self.table[state][action] + self.alpha * (reward + gamma * np.max(self.table[next_state]) - self.table[state][action])
        return self.table[state][action]
    
    def get_row(self, state):
        if(not(isinstance(state, int))):
            state = state[0]

        return self.table[state]

    def get_item(self, state, action):
        return self.table[state][action]
    
class Agent:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.env = env
        
        self.q_table = QTable(alpha, env.observation_space.n, env.action_space.n)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.rewards = []
        self.losses = []
        self.successes = []

    def choose_action(self, state):
        if(random.uniform(0, 1) < self.epsilon):
            return env.action_space.sample()
        else:
            return np.argmax(self.q_table.get_row(state))
        
    def update_table(self, state, action, reward, next_state):
        q = self.q_table.get_item(state, action)
        new_q = self.q_table.update(state, action, reward, next_state, self.gamma)
        loss = (new_q - q) ** 2
        return loss

    def train(self, episodes):
        for e in range(0, episodes):
            state = self.env.reset()
            done = False

            temp_reward = 0
            temp_loss = 0
            steps = 0
            success = 0

            while(not done):
                action = self.choose_action(state)

                next_state, reward, done, truncated, _ = self.env.step(action)
                
                if(not(isinstance(state, int))):
                    state = state[0]
                
                if(not(isinstance(next_state, int))):
                    next_state = next_state[0]
                
                temp_reward += reward
                temp_loss += self.update_table(state, action, reward, next_state)
                state = next_state
                steps += 1

            if not truncated:
                success = 1

            self.rewards.append(temp_reward)
            self.losses.append(float(temp_loss / steps))
            self.successes.append(success)

            if(self.epsilon > self.epsilon_min):
                self.epsilon *= self.epsilon_decay
            
            print(f"QLT - Episode: {e}, \tReward: {temp_reward}, \t\tε: {self.epsilon}")

    def plot_rewards(self):
        window = 10
        plt.figure(figsize = (10, 5))
        plt.plot(self.rewards, alpha = 0.3, color = 'green', label = "Raw Reward")
        plt.plot(np.convolve(self.rewards, np.ones(window), 'valid') / window, color = 'green', label = "Smoothed Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward Curve for Tabular Q-Learning")
        plt.savefig("qlt_rewards.png")

    def plot_loss(self):
        window = 10
        plt.figure(figsize = (10, 5))
        plt.plot(self.losses, alpha = 0.3, color = 'blue', label = "Raw Losses")
        plt.plot(np.convolve(self.losses, np.ones(window), 'valid') / window, label = "Smoothed Accuracy")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Loss Curve for Tabular Q-Learning")
        plt.savefig("qlt_loss.png")

    def plot_accuracy(self):
        window = 10
        accuracy = []
        tmp = 0

        for idx in range(0, len(self.successes)):
            tmp += self.successes[idx]
            accuracy.append(float(tmp / (idx + 1)))

        plt.figure(figsize = (10, 5))
        plt.plot(accuracy, alpha = 0.3, color = 'red', label = "Raw Accuracy")
        plt.plot(np.convolve(accuracy, np.ones(window), 'valid') / window, color = 'red', label = "Smoothed Accuracy")
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve for Tabular Q-Learning")
        plt.savefig("qlt_accuracy.png")

    

env = gym.make('Taxi-v3')
agent = Agent(env, 0.1, 0.99, 1.0, 0.1, 0.995)
episodes = 1000
agent.train(episodes)
agent.plot_rewards()
agent.plot_loss()
agent.plot_accuracy()