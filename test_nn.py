import gym

import torch
import torch.nn as nn
import torch.optim

import numpy as np
import random
from collections import deque

env = gym.make("Taxi-v3")

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128)
        
        def forward(self, par):
            par = torch.relu(self.l1(par))
            par = torch.relu(self.l2(par))
            return self.l3(par)
        
