# MACHINE LEARNING PROJECT - "Taxi" : Tabular Q-Learning vs Single DQN vs Double DQN
# Michele Nicoletti - 1886646
# Lorenzo Pecorari - 1885161

import gymnasium as gym
import numpy as np

# Note: this is an initial file for testing the environment and taking familiarity with everything about it!

class TaxiAgent:
    def __init__(self, learn, e_0, e_decay, e_f, discount):

        self.env = gym.make("Taxi-v3")
        # self.env = gym.make("Taxi-v3", render_mode = "human")

        # rates for q-learning
        self.learning_rate = learn
        self.discount_rate = discount

        # values for epsilon-greedy
        self.init_eps = e_0
        self.final_eps = e_f
        self.eps_decay = e_decay
        self.epsilon = self.init_eps

        # sets to zero all alues into q-learning table on the basis of the number of possible states reachable by the agent
        self.q_vals = np.zeros(self.env.action_space.n)
        
    def first_action(self, observation, reward, terminated, truncated, info):
        observation, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
        print(f"observation: {observation},\n reward: {reward},\n terminated: {terminated},\n truncated: {terminated},\n info : {info}\n")

    # it picks ana action by using the epsilon approach
    def get_action(self, observation, reward, terminated, info):
        random_num = np.random.random()
        print(f"{random_num} vs {self.epsilon} = {random_num < self.epsilon}")
        
        if random_num < self.epsilon:
            self.epsilon -= self.eps_decay
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_vals)

            
    # it updates the q-learning table after the execution of the action
    def update_agent(self, observation, reward, terminated, info):
        pass

    # function for resetting the environment and starting to execute the training of the agent
    def start(self):

        observation, info = self.env.reset(seed=10)
        self.init_eps = 1

        observation = None 
        reward = None
        terminated = None
        truncated = None
        info = None
        action = None
        
        # first action
        self.first_action(observation, reward, terminated, truncated, info)

        # executional cycle
        for _ in range(1024):
            # I need to use an algorithm for picking the right action to execute!
            action = self.get_action(observation, reward, terminated, info) 

            observation, reward, terminated, truncated, info = self.env.step(action)
            print(f"observation: {observation},\n reward: {reward},\n terminated: {terminated},\n info : {info}")
           
            # I need to use the observation for updating the table and taking the next action
            # observation obtained as:
            #   ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
                
            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()

# "main" calls
# REMEMBER : TaxiAgent (learn rate, initial epsilon, epsilon-decade rate, final epsilon, discount rate)
taxi = TaxiAgent(0, 1, 0.005, 0, 0)
taxi.start()