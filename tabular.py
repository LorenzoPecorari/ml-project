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
    
    def get_alpha(self):
        return self.alpha
    
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
            loss = 0
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
                loss = self.update_table(state, action, reward, next_state)
                state = next_state
                steps += 1

            if not truncated:
                success = 1

            self.rewards.append(temp_reward)
            self.losses.append(np.mean(loss))            
            self.successes.append(success)

            if(self.epsilon > self.epsilon_min):
                self.epsilon *= self.epsilon_decay
            
            print(f"QLT - Episode: {e}, \tReward: {temp_reward}, \t\tε: {self.epsilon}")

    def plot_rewards(self):
        window = 10
        plt.figure(figsize = (10, 5))
        plt.plot(self.rewards, alpha = 0.3, color = 'green', label = "Raw Reward")
        plt.plot(range(window - 1, len(self.rewards)), np.convolve(self.rewards, np.ones(window) / window, mode = 'valid'), label = "Smoothed Rewards", color = 'green')
        
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.suptitle("Reward Curve for Tabular Q-Learning")
        plt.title(f"α = {'%.4f'%self.q_table.get_alpha()}, γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%self.epsilon}, ε_min = {'%.4f'%self.epsilon_min}, ε_decay = {'%.4f'%self.epsilon_decay}")
        
        plt.savefig("qlt_rewards.png")
        plt.clf()

    def plot_loss(self):
        window = 10
        plt.figure(figsize = (10, 5))
        
        plt.plot(self.losses, alpha = 0.3, color = 'blue', label = "Raw Loss")
        plt.plot(range(window - 1, len(self.losses)), np.convolve(self.losses, np.ones(window) / window, mode = 'valid'), label = "Smoothed Loss", color = 'blue')
        
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.suptitle("Loss Curve for Tabular Q-Learning")
        plt.title(f"α = {'%.4f'%self.q_table.get_alpha()}, γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%self.epsilon}, ε_min = {'%.4f'%self.epsilon_min}, ε_decay = {'%.4f'%self.epsilon_decay}")
        
        plt.savefig("qlt_loss.png")
        plt.clf()

    def plot_accuracy(self):
        window = 10
        accuracy = []
        tmp = 0

        for idx in range(0, len(self.successes)):
            tmp += self.successes[idx]
            accuracy.append(float(tmp / (idx + 1)))

        plt.figure(figsize = (10, 5))
        plt.plot(accuracy, alpha = 0.3, color = 'red', label = "Raw Accuracy")
        plt.plot(range(window - 1, len(accuracy)), np.convolve(accuracy, np.ones(window) / window, mode = 'valid'), label = "Smoothed Accuracy", color = 'red')

        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.suptitle("Accuracy Curve for Tabular Q-Learning")
        plt.title(f"α = {'%.4f'%self.q_table.get_alpha()}, γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%self.epsilon}, ε_min = {'%.4f'%self.epsilon_min}, ε_decay = {'%.4f'%self.epsilon_decay}")

        plt.savefig("qlt_accuracy.png")
        plt.clf()

if __name__ == "__main__":    
    lr = 0.0005
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.998
    episodes = 8000
    
    env = gym.make("Taxi-v3")
    agent = Agent(env, lr, gamma, epsilon, epsilon_min, epsilon_decay)
    
    agent.train(episodes)
    agent.plot_rewards()
    agent.plot_loss()
    agent.plot_accuracy()