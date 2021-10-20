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

#------------- add
from utils.myutils import weight_init
from utils.myutils import fanin_init
from torch.distributions import Categorical
from torch.autograd import Variable
from torch import Tensor
def insert_action(x):
    action_array = [0 for i in range(len(actions))]
    action_array[x]=1
    return action_array

HIDDEN_DIM = 300
actions = ['N','S','E','W','NW','WS','SE','EN','.']
idx_to_act = {0:"N",1:"S",2:"E",3:"W", 4:"NW",5:"WS",6:"SE",7:"EN",8:"."}
cast = lambda x: Variable(Tensor(x).cuda(), requires_grad=False)

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
        self.fc4 = nn.Linear(32, 9)
        
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)

    def forward(self, state_img, state_val):

        print('state_img',state_img.size())
        x = self.conv1(state_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc_val(state_val))
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.fc2(x))
        print('this is feature', x,
              'size', x.size())
        
        z = F.relu(self.fc5(x))
        z = self.fc6(z)

        x = F.relu(self.fc3(x))
        x = self.fc4(x) # actor
        
        return F.log_softmax(x, dim=1), z


class Actor(nn.Module):
    def __init__(self,number_of_agents):
        super(Actor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # 32 * 2 * 2

        # 128 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(32 * 2 * 2, 64)
        self.fc_val = nn.Linear(3, 64)
        self.fc2 = nn.Linear(128, 64)


# add
        self.s_dim = 64
        self.a_dim = 9
        self.n_agents = number_of_agents

        # input (batch, s_dim) output (batch, 300)
        self.prev_dense = DenseNet(self.s_dim, HIDDEN_DIM, HIDDEN_DIM // 2, output_activation=None, norm_in=True)
        # input (num_agents, batch, 200) output (num_agents, batch, num_agents * 2)\
        self.comm_net = LSTMNet(HIDDEN_DIM // 2, HIDDEN_DIM // 2, num_layers=1)
        # input (batch, 2) output (batch, s_dim)
        self.post_dense = DenseNet(HIDDEN_DIM + self.s_dim, HIDDEN_DIM // 2, self.a_dim, output_activation=nn.Tanh)
# ----------------
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 9)

    def forward(self, state_img, state_val):
        state_img = torch.tensor(state_img)
        state_val = torch.tensor(state_val)
        state_img = state_img.float().to(self.device)
        state_val = state_val.float().to(self.device)
        x = self.conv1(state_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        img_list = x.data.cpu().numpy()
        img_list=[torch.tensor([img]).cuda() for img in img_list]
        # img_list = [x]
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc_val(state_val))
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.fc2(x))
        x_s = torch.unsqueeze(x,-1).view(1, self.n_agents, self.s_dim)
        x = self.prev_dense(x)
        x = x.reshape(-1, self.n_agents, HIDDEN_DIM // 2)
        x = self.comm_net(x)

        x = torch.cat((x, x_s), dim=-1)
        x = self.post_dense(x)
        x = x.view(-1, self.n_agents, self.a_dim)
        # print('x',x)

        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)  # actor
        probs = F.log_softmax(x, dim=1)
        actions_probs = F.softmax(x, dim=1)

        probs = torch.exp(probs)
        m = Categorical(probs)
        _, greedy_action = torch.max(probs.data, 1)
        origin_actions = m.sample()

        action_list = []
        attion_actions_list = []

        for i in origin_actions.cpu().numpy()[0]:
            # print(i.cpu().data())
            # print('idx_to_act[i]', idx_to_act[i])
            action_list.append(idx_to_act[i])
            attion_actions_list.append(cast([insert_action(i)]))
        # add--------------------
        return attion_actions_list, action_list, img_list, actions_probs, m.log_prob(origin_actions), m.entropy()

class DenseNet(nn.Module):
    def __init__(self, s_dim, hidden_dim, a_dim, norm_in=False, hidden_activation=nn.ReLU, output_activation=None):
        super(DenseNet, self).__init__()

        self._norm_in = norm_in

        if self._norm_in:
            self.norm1 = nn.BatchNorm1d(s_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
            self.norm3 = nn.BatchNorm1d(hidden_dim)
            self.norm4 = nn.BatchNorm1d(hidden_dim)

        self.dense1 = nn.Linear(s_dim, hidden_dim)
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2.weight.data = fanin_init(self.dense2.weight.data.size())
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3.weight.data.uniform_(-1, 1)
        self.dense4 = nn.Linear(hidden_dim, a_dim)

        if hidden_activation:
            self.hidden_activation = hidden_activation()
        else:
            self.hidden_activation = lambda x : x

        if output_activation:
            self.output_activation = output_activation()
        else:
            self.output_activation = lambda x : x

    def forward(self, x):
        use_norm = True if (self._norm_in and x.shape[0] != 1) else False

        if use_norm: x = self.norm1(x)
        x = self.hidden_activation(self.dense1(x))
        if use_norm: x = self.norm2(x)
        x = self.hidden_activation(self.dense2(x))
        if use_norm: x = self.norm3(x)
        x = self.hidden_activation(self.dense3(x))
        if use_norm: x = self.norm4(x)
        x = self.output_activation(self.dense4(x))
        return x


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1,
                 bias=True,
                 batch_fisrt=True,
                 bidirectional=True):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_fisrt,
            bidirectional=bidirectional
        )

    def forward(self, input, wh=None, wc=None):
        output, (hidden, cell) = self.lstm(input)
        return output