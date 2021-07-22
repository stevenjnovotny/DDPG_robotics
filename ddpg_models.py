import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lecun:  initialize hidden layers as a function of fan-in (Xavier initialization)
# Fan-in: the maximum number of inputs that a system can accept
# Bengio: Biases can generally be initialized to zero but weights need to be initialized 
# -- carefully to break the symmetry between hidden units of the same layer.
def xav_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)



class Critic(nn.Module):
    """ Critic Model --- to learn Value function """

    def __init__(self, state_size, action_size, seed, fc1_size=256, fc2_size=128, fc3_size=128):
        # 1 agent: fc1_size=256, fc2_size=128
        """
        params:
            state_size (int)
            action_size (int)
            seed(int)
            fc1_size(int)
            fc2_size(int)
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size + action_size, fc2_size)  # add in actions for DDPG
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, 1)  # gives single Q-value 
        self.reset_weights()

    def reset_weights(self):
        # self.fc1.weight.data.uniform_(*xav_init(self.fc1))
        # self.fc2.weight.data.uniform_(*xav_init(self.fc2))
        # self.fc3.weight.data.uniform_(*xav_init(self.fc3))  
        # self.fc4.weight.data.uniform_(-3e-2, 3e-2)  # to keep initial outputs from policy around zero

        torch.nn.init.xavier_uniform_(self.fc1.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
        torch.nn.init.xavier_uniform_(self.fc2.weight.data, gain=nn.init.calculate_gain('leaky_relu'))      
        torch.nn.init.xavier_uniform_(self.fc3.weight.data, gain=nn.init.calculate_gain('leaky_relu'))  
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """ maps (state, action) to Q-values """
        state = self.bn(state)
        xs = F.leaky_relu(self.fc1(state))  # leaky is supposed to speed up training
        x = torch.cat((xs,action), dim=1) # add in action
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)


class Actor(nn.Module):
    """ Actor Model --- to learn Policty function """

    def __init__(self, state_size, action_size, seed, fc1_size=256, fc2_size=128):
        # 1 agent: fc1_size=128
        """
        params:
            state_size (int)
            action_size (int)
            seed(int)
            fc1_size(int)
            fc2_size(int) -- eliminated to help with training on 1 actor
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)
        #self.fc2 = nn.Linear(fc1_size, action_size)
        self.reset_weights()

    def reset_weights(self):
        # self.fc1.weight.data.uniform_(*xav_init(self.fc1))
        # self.fc2.weight.data.uniform_(*xav_init(self.fc2))
        # self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        torch.nn.init.xavier_uniform_(self.fc1.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
        torch.nn.init.xavier_uniform_(self.fc2.weight.data, gain=nn.init.calculate_gain('leaky_relu'))      
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ maps state to action """
        state = self.bn(state)
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        #x = F.tanh(self.fc2(x))
        return x


if __name__ == "__main__":
    print("**** Testing integrity of networks... ****")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    a = Actor(30,4,1234)
    c = Critic(30,4,1234)

    states = np.array([np.random.random(30), np.random.random(30)])
    states = torch.from_numpy(states).float().unsqueeze(0).to(device)
    print(states)
    action = a.forward(states)
    print(action)
    q = c.forward(states,action)
    print(q)
