import argparse
import gym
from gym import wrappers
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from envs import make_retro

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')

args = parser.parse_args()

env = make_retro('SonicTheHedgehog-Genesis', 'LabyrinthZone.Act1')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

actions = [[1,0,0,0,0,0,0,0,0,0,0,0], # b
        [0,0,0,0,0,0,0,1,0,0,0,0], # right
        [0,0,0,0,0,0,1,0,0,0,0,0], # left
        [0,0,0,0,0,1,0,0,0,0,0,0], # down
        [0,0,0,0,0,1,1,0,0,0,0,0], # left down
        [0,0,0,0,0,1,0,1,0,0,0,0], # right down
        [1,0,0,0,0,1,0,0,0,0,0,0]] # down b

class Policy(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.affine1 = nn.Linear(32*3*11, 256)
        self.action_head = nn.Linear(256, action_space)
        self.value_head = nn.Linear(256, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32*3*11)
        x = F.elu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


model = Policy(env.observation_space.shape[0], 7)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(Variable(state))
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.data[0]

model.load_state_dict(torch.load('weights/{}.pt'.format("actor_critic_sonic1")))

running_length = 10
max_reward = -100
for i_episode in count(1):
    state = env.reset()
    current_reward = 0
    done = False
    t = 0
    while not done:
        action = actions[int(select_action(np.array(state)))]
        state, reward, done, _ = env.step(action)
        env.render()
        model.rewards.append(reward)
        current_reward+=reward
        t+=1

    running_length = running_length * 0.99 + t * 0.01
    print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}\tReward: {:.5f}'.format(
        i_episode, t, running_length, current_reward))