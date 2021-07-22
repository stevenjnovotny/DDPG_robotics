from unityagents import UnityEnvironment
import numpy as np

"""
Set up unity ML-agents environment
"""

# env = UnityEnvironment(file_name='Reacher.app')
env = UnityEnvironment(file_name='Reacher_multi.app', no_graphics=True)
#env = UnityEnvironment(file_name='Reacher_multi.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

"""
Examine state and action space

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is 
provided for each step that the agent's hand is in the goal location. Thus, the goal of the 
agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, 
and angular velocities of the arm. Each action is a vector with four numbers, corresponding to 
torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1.

"""

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


"""
Train an agent using DDPG

"""

from ddpg_agent import Agent
from collections import deque
import torch
import time

# create an agent
seed = 42
agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed, Nagents=num_agents)

n_episodes = 500
max_t = 1000
print_every = 10
window = 100 # for calculating success
success = 30

scores_deque = deque(maxlen=window)
score_data = []

for i_episodes in range(1, n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    agent.reset()
    scores = np.zeros(num_agents) 
    for t in range(max_t):
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        agent.step(states, actions, rewards, next_states, dones)
        scores += np.array(rewards)   
        states = next_states
        if np.any(dones):
            break

    avg_score = np.mean(scores)
    score_data.append(avg_score)
    scores_deque.append(avg_score)
    avgWin = np.mean(scores_deque) # calc best score over last window

    print('\rEpisode: {}  \tEpisode Average: {:.1f} \tRolling Avereage:  {:.3f}'.format(i_episodes, avg_score, avgWin), end="")
    if i_episodes % print_every == 0:
        print('\rEpisode: {}  \tEpisode Average: {:.2f} \tRolling Avereage:  {:.3f}'.format(i_episodes, avg_score, avgWin))
    if avgWin >= success:
        print('\nEnvironment solved')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        break

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(score_data)+1), score_data)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scoresVepisodes.png')
plt.show()