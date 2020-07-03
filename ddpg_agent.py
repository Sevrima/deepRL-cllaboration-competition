
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Actor, Critic
import random
import copy
from collections import namedtuple, deque

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.90            # discount factor
TAU = .999              # for soft update of target parameters
LR_ACTOR = 1e-4               
LR_CRITIC = 1e-4        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, state_size, action_size, seed):
        """Initializing the agent"""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optim = optim.Adam( self.critic_local.parameters(), lr = LR_CRITIC)


        self.noise = OUNoise(action_size, seed = self.seed)
        self.buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed = self.seed) 

    def act(self, state, add_noise = True):
        """returning actions from the current policy"""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions,-1,+1)
    
    def reset(self):
        self.noise.reset()

    def step(self, states, actions, rewards, next_states, dones ):
        """saves experiences in buffer"""
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.buffer.add(state, action, reward, next_state, done)
        self.start_learn()

    def start_learn(self):
        """calls the learn method"""    
        if len(self.buffer) > BATCH_SIZE:
            experiences = self.buffer.sample()
            self.learn(experiences, GAMMA)

        

    def learn(self, experiences, gamma):
        """updates policy and value networks given a batch of experiences"""
        states, actions, rewards, next_states, dones = experiences

        #updating Critic_local
        next_actions  = self.actor_target(next_states)
        next_Q_target = self.critic_target(next_states, next_actions)
        Q_target = rewards + gamma*(1-dones)*next_Q_target
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_target, Q_expected)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()


        #updating Actor_local
        next_actions = self.actor_local(states)
        actor_loss = - self.critic_local(states, next_actions).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
    
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)    

    def soft_update(self, local_network, target_network, tau):
        """updates target network using polyak averaging"""
        for local_parameter, target_parameter in zip(local_network.parameters(), target_network.parameters()):
            target_parameter.data.copy_((1.0-tau)*local_parameter+tau*target_parameter)
        

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state

        # Thanks to Hiu C. for this tip, this really helped get the learning up to the desired levels
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)

        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)