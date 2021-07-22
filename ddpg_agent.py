from math import exp, gamma, tau
from os import EX_NOPERM
import numpy as np 
import random
import copy
from collections import namedtuple, deque

from ddpg_models import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)      # replay buffer size
BATCH_SIZE = 1024           # minibatch size  -- 64cfor 1-agent
GAMMA = 0.99                # discount factor
TAU = 1e-3                  # interpolation parameter for ddpg soft update
LR_ACTOR = 1e-4            # learning rate for actor
LR_CRITIC = 3.0e-4           # learning rate for critic
WEIGHT_DECAY = 0.000           # L2 weight decay
LEARN_EVERY = 20            # how many time steps between learning
UPDATE_TIMES = 10           # number or times to update each learning event
# EPSILON = 1.0               # exploration/exploitation parameter via noise
# EPSILON_DECAY = 1e-6        # decay rate for noise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ ddpg agent that interacts with an environment """

    def __init__(self, state_size, action_size, random_seed=0, Nagents = 1):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        #self.epsilon = EPSILON
        self.Nagents = Nagents

        # initialize critic (local and target network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # initialize actor (local and target network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # create a noise model
        self.noise = OUNoise(action_size, random_seed)

        # set replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # set time step counter for learning
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        """ 
        save experience (for each agent) to replay memory and use random sampling to learn
        """
        for i in range(self.Nagents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        self.t_step += 1
        if self.t_step % LEARN_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                for _ in range(UPDATE_TIMES):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
                    #self.t_step = 0

    def act(self, states, add_noise=True, train=True):
        """ returns actions for a given state based on current policy """
        #state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        if train==True:
            self.actor_local.train()
        if add_noise:
            #action += self.epsilon * self.noise.sample()
            noise = [self.noise.sample() for i in range(self.Nagents)]
            actions += noise
            #actions += self.noise.sample()

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """ update polciy and value estimators
            -- using batch of experiecne tuples
            -- Q_target = r + gamma * critic_target(next_state, actor_target(next_state)) 
            -- recall
            ---- actor_target(state) -> action  
            ---- critic_traget(state, action) -> Q_value

            expereinces (Tuple[torch.tensor]): tuple of (s,a,r,s',done) tuples
            gamma (float): discount parameter
        """

        states, actions, rewards, next_states, dones  = experiences
        
        # --- update critic --- #
        # get predicted actions for next_states and Q-values -- all from target models
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        # compute new Q values for current state y_i
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        # compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)  # extending pytorch computational graph for critic_local
        # minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()      # updates param.grad for critic_local
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optimizer.step()        # for local critic


        # --- update actor ---- #
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # negative sign for gradient ascent; 
        # J = expectation(Q(s,a)) = actor_loss
        # we want the actor to give action that maximizes the Q_value. So the actor's error is (- Q_value)

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- update target networks --- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # --- update noise via epsilon greedy ---- #
        # self.epsilon -= EPSILON_DECAY
        # self.noise.reset()


    def soft_update(self, local_model, target_model, tau):
        """ Weights_target = tau * Weights_local + (1-tau) * Weights_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0-tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.random() for i in range(len(x))])
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.normal() for i in range(len(x))])

        self.state = x + dx
        return self.state


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Paramaters:
            - buffer_size (int) : max size of buffer
            - batch_size (int) : size of each training batch
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ Randomly sample a batch of experiences from memory """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)    
         

if __name__ == "__main__":
    print("**** testing integrity of agent and supporting classes **** ")
    rp = ReplayBuffer(4, BUFFER_SIZE, BATCH_SIZE, 1234)
    print(len(rp))
