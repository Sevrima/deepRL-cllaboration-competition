import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1_units =  256, fc2_units = 256, fc3_units = 64):
        super().__init__()

        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return torch.tanh(self.fc4(x))




class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1_units =  256, fc2_units = 128, fc3_units = 32):
        super().__init__()

        self.fc1 = nn.Linear(state_size+action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

    def forward(self, state,action):
        x = torch.cat([state.float(), action.float()], 1)
        # x = state+action
        x = self.fc1(x.float())
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return self.fc4(x)
        