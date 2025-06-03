import gymnasium as gym
import random

import numpy as np

import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
class QTable:
    def __init__(self, states, actions):
        self.table = np.zeros((states, actions))

    def update(self, state, action, gamma, next_state, reward, alpha):
        next_states = env.unwrapped.P[state][action]

        self.table[state][action]=(1-alpha)*self.table[state][action]+ alpha*(reward+ gamma*np.max(self.table[next_state]))

        return self.table[state][action]
    
    def get_row(self, state):
        if(not(isinstance(state, int))):
            state = state[0]

        return self.table[state]
    
class Agent:
    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_decay, alpha):
        self.env = env
        
        self.q_table = QTable(env.observation_space.n, env.action_space.n)
        
        self.gamma = gamma
        self.alpha = alpha
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
        
    def update_table(self, state, action, next_state, reward):
        self.q_table.update(state, action, self.gamma, next_state, reward, alpha)

    def train(self, episodes):
        for e in range(0, episodes):
            state = self.env.reset()
            done = False

            temp_reward = 0
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
                self.update_table(state, action, next_state, reward)
                state = next_state
                steps += 1

            if not truncated:
                success = 1

            self.rewards.append(temp_reward)
            self.successes.append(success)

            if(self.epsilon > self.epsilon_min):
                self.epsilon *= self.epsilon_decay
            
            print(f"QLT - Episode: {e}, \tReward: {temp_reward}, \t\tε: {self.epsilon}")

    def get_rewards(self):
        return self.rewards
    
    def get_successes(self):
        return self.successes

    def plot_rewards(self):
        window = 10
        plt.figure(figsize = (10, 5))
        plt.plot(self.rewards, color = 'green', label = "Raw Reward")
        # plt.plot(range(window - 1, len(self.rewards)), np.convolve(self.rewards, np.ones(window) / window, mode = 'valid'), label = "Smoothed Rewards")
        
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.suptitle("Reward Curve for Tabular Q-Learning")
        # plt.title(f"γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%self.epsilon}, ε_min = {'%.4f'%self.epsilon_min}, ε_decay = {'%.4f'%self.epsilon_decay}")
        
        plt.show(block= False)
        plt.savefig(f"./plots/qlt_rewards_{str(self.epsilon_decay * 100)}_{str(self.gamma * 100)}.pdf.pdf")
        # plt.clf()

    def plot_accuracy(self):
        window = 10
        accuracy = []
        tmp = 0

        for idx in range(0, len(self.successes)):
            tmp += self.successes[idx]
            accuracy.append(float(tmp / (idx + 1)))

        plt.figure(figsize = (10, 5))
        plt.plot(accuracy, color = 'red', label = "Raw Accuracy")
        # plt.plot(range(window - 1, len(accuracy)), np.convolve(accuracy, np.ones(window) / window, mode = 'valid'), label = "Smoothed Accuracy")

        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.suptitle("Accuracy Curve for Tabular Q-Learning")
        plt.title(f"γ = {'%.4f'%(self.gamma)}, ε = {'%.4f'%self.epsilon}, ε_min = {'%.4f'%self.epsilon_min}, ε_decay = {'%.4f'%self.epsilon_decay}")

        plt.show(block= False)
        plt.savefig(f"./plots/qlt_accuracy_{str(self.epsilon_decay * 100)}_{str(self.gamma * 100)}.pdf")
        # plt.clf()

def plot_rewards(structure, filename):
    window = 10
    plt.figure(figsize=(10, 5))
    for elemento in structure:
        plt.plot(elemento[1], label=f"{elemento[0]} - raw")
        # plt.plot(
            # range(window - 1, len(elemento[1])),
            # np.convolve(elemento[1], np.ones(window) / window, mode='valid'),
            # label=f"{elemento[0]} - smooth"
        # )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.suptitle("Rewards Curve for Tabular Q-Learning")
    plt.legend()
    plt.grid()
    # plt.ylim(-500, 20)
    plt.savefig(f"{filename}_const_rewards.pdf")


def plot_accuracies(structure, filename):
    window = 10
    plt.figure(figsize=(10, 5))
    for elemento in structure:
        i = 0
        tmp = 0
        accuracy = []
        for elem in elemento[1]:
            tmp += elem
            accuracy.append(float(tmp / (i + 1)))
            i += 1
        
        plt.plot(accuracy, label=f"{elemento[0]} - raw")
        
            # plt.plot(accuracy, alpha = 0.3, label=f"Run {structure.index(elemento)} - raw")
            # plt.plot(
            #     range(window - 1, len(accuracy)),
            #     np.convolve(accuracy, np.ones(window) / window, mode='valid'),
            #     label=f"Run {structure.index(elemento)}"
            # )
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.suptitle("Accuracy Curve for Tabular Q-Learning")
    plt.legend()
    plt.grid()
    plt.savefig(f"{filename}_const_accuracy.pdf")

def plot_rewards_zoom(structure, filename):
    window = 10
    plt.figure(figsize=(10, 5))
    for elemento in structure:
        # plt.plot(elemento[1], label=f"{elemento[0]} - raw")
        plt.plot(
            range(window - 1, len(elemento[1])),
            np.convolve(elemento[1], np.ones(window) / window, mode='valid'),
            label=f"{elemento[0]} - smooth"
        )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.suptitle("Rewards Curve for Tabular Q-Learning")
    plt.legend()
    plt.grid()
    plt.ylim(-500, 20)
    plt.savefig(f"{filename}_const_rewards_zoom.pdf")

if __name__ == "__main__":

    # env = gym.make("Taxi-v3")
    rewards = []
    accuracies = []

    ### TESTS FOR CONSTANT GAMMA 

    episodes = 1000
    alpha=0.1
    gamma = 0.5
    epsilon = 1.0
    epsilon_min = 0.1

    epsilon_decay = 0.998
    agent = Agent(env, gamma, epsilon, epsilon_min, epsilon_decay, alpha)
    agent.train(episodes)    
    rewards.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_rewards()])
    accuracies.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_successes()])
 
    # epsilon_decay = 0.995
    # agent = Agent(env, gamma, epsilon, epsilon_min, epsilon_decay)    
    # agent.train(episodes)
    # rewards.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_rewards()])
    # accuracies.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_successes()])

    epsilon_decay = 0.95
    agent = Agent(env, gamma, epsilon, epsilon_min, epsilon_decay, alpha)    
    agent.train(episodes)
    rewards.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_rewards()])
    accuracies.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_successes()])

    # epsilon_decay = 0.9
    # agent = Agent(env, gamma, epsilon, epsilon_min, epsilon_decay)
    # agent.train(episodes)
    # rewards.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_rewards()])
    # accuracies.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_successes()])

    epsilon_decay = 0.85
    agent = Agent(env, gamma, epsilon, epsilon_min, epsilon_decay, alpha)
    agent.train(episodes)
    rewards.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_rewards()])
    accuracies.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_successes()])

    plot_rewards(rewards, f"γ_{agent.gamma}")
    plot_accuracies(accuracies, f"γ_{agent.gamma}")
    plot_rewards_zoom(rewards, f"γ_{agent.gamma}")

    ### TESTS FOR CONSTANT E_DECs
    # epsilon_decay = 0.998
    # agent = Agent(env, gamma, epsilon, epsilon_min, epsilon_decay)
    # agent.train(episodes)    
    # rewards.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_rewards()])
    # accuracies.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_successes()])

    # epsilon_decay = 0.995
    # agent = Agent(env, gamma, epsilon, epsilon_min, epsilon_decay)    
    # agent.train(episodes)
    # rewards.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_rewards()])
    # accuracies.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_successes()])

    # gamma = 0.9

    # epsilon_decay = 0.95   
    # agent = Agent(env, gamma, epsilon, epsilon_min, epsilon_decay)    
    # agent.train(episodes)
    # rewards.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_rewards()])
    # accuracies.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_successes()])

    # epsilon_decay = 0.9
    # agent = Agent(env, gamma, epsilon, epsilon_min, epsilon_decay)
    # agent.train(episodes)
    # rewards.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_rewards()])
    # accuracies.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_successes()])

    # epsilon_decay = 0.85
    # agent = Agent(env, gamma, epsilon, epsilon_min, epsilon_decay)
    # agent.train(episodes)
    # rewards.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_rewards()])
    # accuracies.append([f"γ: {agent.gamma}, ε_dec: {agent.epsilon_decay}", agent.get_successes()])

    # plot_rewards(rewards, f"γ_{agent.gamma}")
    # plot_accuracies(accuracies, f"γ_{agent.gamma}")
    # plot_rewards_zoom(accuracies, f"γ_{agent.gamma}")