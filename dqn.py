#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:45:26 2020

@author: simon, flo, tim
"""


import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from resnet import ResNet

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, fc5_dims, fc6_dims, fc7_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims= fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.fc6_dims = fc6_dims
        self.fc7_dims = fc7_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims,self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims,self.fc5_dims)
        self.fc6 = nn.Linear(self.fc5_dims,self.fc6_dims)
        self.fc7 = nn.Linear(self.fc6_dims,self.fc7_dims)
        self.fc8 = nn.Linear(self.fc7_dims,self.n_actions)
        self.dp1 = T.nn.Dropout(0.5)
        self.dp2 = T.nn.Dropout(0.3)
        self.dp3 = T.nn.Dropout(0.2)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
    def forward(self, state):
        x = F.relu(self.fc1(state.float()))
        x = F.relu(self.fc2(x))
        x = self.dp1(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dp2(x)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.dp3(x)
        x = F.relu(self.fc7(x))
        actions = self.fc8(x)
        return actions
    
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, eps_dec, max_mem_size=50000, eps_end=0.01):
        self.gamma = gamma 
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size 
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(self.lr, input_dims=input_dims, fc1_dims=256, fc2_dims=128, fc3_dims=64, fc4_dims=64, fc5_dims=32, fc6_dims=32, fc7_dims=16, n_actions=n_actions)
        #self.Q_eval = ResNet()
        
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype= np.bool)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def choose_action(self,observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
            
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return


        for i in range(30):
            self.Q_eval.optimizer.zero_grad()
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
            reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

            action_batch = self.action_memory[batch]
            #print(f"State: {(state_batch.view(state_batch.size()[0], state_batch.size()[1], 1, 1)).size()}")
            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
            #q_eval = self.Q_eval.forward(state_batch.view(6,7,1))[batch_index, action_batch]
            q_next = self.Q_eval.forward(new_state_batch)
            q_next[terminal_batch] = 0.0

            q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

            
    def save_checkpoint(self):
        state = {'state_dict': self.Q_eval.state_dict(), 'optimizer': self.Q_eval.optimizer.state_dict(), 'state_memory': self.state_memory,
                 'new_state_memory': self.new_state_memory, 'reward_memory': self.reward_memory, 'action_memory': self.action_memory,
                 'terminal_memory': self.terminal_memory, 'mem_cntr': self.mem_cntr}
        T.save(state, 'dqn_network')
        return
        
    def load_checkpoint(self,filename='dqn_network'):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = T.load(filename)
            self.Q_eval.load_state_dict(checkpoint['state_dict'])
            self.Q_eval.optimizer.load_state_dict(checkpoint['optimizer'])
            self.state_memory = checkpoint['state_memory']
            self.new_state_memory = checkpoint['new_state_memory']
            self.reward_memory = checkpoint['reward_memory']
            self.action_memory = checkpoint['action_memory']
            self.terminal_memory = checkpoint['terminal_memory']
            self.mem_cntr = checkpoint['mem_cntr']
            print("=> loaded checkpoint '{}'".format(filename))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
        return 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        