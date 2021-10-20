'''
@Author: Baiming Chen, Zuxin Liu
@Email: {baimingc, zuxinl}@andrew.cmu.edu
@Date:   2020-02-19 21:28:23
@LastEditTime: 2020-03-26 00:40:17
@Description:
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.autograd import Variable
from torch import  Tensor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0) #32 * 2 * 2
        
        #128 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(32*2*2, 64)
        self.fc_val = nn.Linear(3, 64)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 9) #actor
        
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)

    def forward(self, state_img, state_val):
        x = self.conv1(state_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        k=x
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc_val(state_val))
        x = torch.cat((x, y), dim=1)

        x = F.relu(self.fc2(x))
        
        z = F.relu(self.fc5(x))
        z = self.fc6(z)

        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        

        return k, F.log_softmax(x, dim=1),F.softmax(x,dim=1)   #return log_prob and value




