# MACHINE LEARNING PROJECT - "Taxi" : Tabular Q-Learning vs Single DQN vs Double DQN
# Michele Nicoletti - 1886646
# Lorenzo Pecorari - 1885161

import gymnasium as gym
import numpy as np

# Note: this is an initial file for testing the environment and taking familiarity with everything about it!

class TaxiAgent:
    def __init__(self, learn, e_0, e_decay, e_f, discount):
        
        self.env = gym.make("Taxi-v3", render_mode = "human")

        # rates for q-learning
        self.learning_rate = learn
        self.discount_rate = discount

        # values for epsilon approach
        self.init_eps = e_0
        self.final_eps = e_f
        self.eps_decay = e_decay
        self.epsilon = self.init_eps

        # sets to zero all alues into q-learning table on the basis of the number of possible states reachable by the agent
        self.q_vals = np.zeros(self.env.action_space.n)

    # it picks ana action by using the epsilon approach
    def get_action(self, observation, reward, terminated, info):
        if(np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        
        else:
            return self.q_vals.index(max(self.q_vals))
            
    # it updates the q-learning table after the execution of the action
    def update_agent(self, observation, reward, terminated, info):
        pass

    # function for resetting the environment and starting to execute the training of the agent
    def start(self):

        observation, info = self.env.reset(seed=10)
        self.init_eps = 1

        # cycles of 
        for _ in range(1024):
            # I need to use an algorithm for picking the right action to execute!
            # action = self.get_action(observation) # <- to be commented until methods not completely implemented 

            observation, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            print(f"observation: {observation},\n reward: {reward},\n terminated: {terminated},\n info : {info}")
           
            # observation obtained as:
            #   ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
                
            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()

# "main" calls
taxi = TaxiAgent(0, 0, 0, 0, 0)
taxi.start()