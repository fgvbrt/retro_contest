import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import utils


class NatureCNN(nn.Module):
    def __init__(self, n_ac, ob_space):
        super(NatureCNN, self).__init__()
        nc, nh, nw = ob_space.shape
        out_h = utils.convs_out_dim(nh, [8, 4, 3], [0, 0, 0], [4, 2, 1])
        out_w = utils.convs_out_dim(nw, [8, 4, 3], [0, 0, 0], [4, 2, 1])
        self.conv_out = out_w * out_h * 64

        self.conv1 = nn.Conv2d(nc, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.conv_out, 512)
        self.action_head = nn.Linear(512, n_ac)
        self.value_head = nn.Linear(512, 1)

    def forward(self, inp):
        x = F.relu(self.conv1(inp))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.conv_out)
        x = F.relu(self.fc1(x))

        act_logits = F.log_softmax(self.action_head(x), -1)
        vals = self.value_head(x)

        return act_logits, vals


class CNNPolicy(object):
    def __init__(self, ob_space, ac_space, vf_coef, ent_coef, lr):
        self.model = NatureCNN(ac_space.n, ob_space)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.loss_names = ("pg_loss", "entropy", "vf_loss", "clipfrac", "approxkl")

    def value(self, obs):
        act_logits, vals = self.model(obs)
        return vals

    def step(self, obs, sample=True):
        # make batch of one
        # obs = torch.tensor(obs).unsqueeze(0)
        obs = torch.from_numpy(obs).unsqueeze(0)
        act_logits, vals = self.model(obs)

        # escape from batch
        act_logits = act_logits[0]
        vals = vals[0]

        if sample:
            action = Categorical(logits=act_logits).sample()
        else:
            action = act_logits.argmax()

        return action, act_logits[action], vals

    def train(self, cliprange, obs, advs, returns, actions, old_vals, old_logits):

        # TODO: may store all of it in tensors already
        obs = torch.from_numpy(obs)
        actions = torch.from_numpy(actions)
        old_vals = torch.from_numpy(old_vals)
        old_logits = torch.from_numpy(old_logits)
        advs = torch.from_numpy(advs)
        returns = torch.from_numpy(returns)

        # current logits and vals
        act_logits, vals = self.model(obs)
        logits = act_logits.gather(1, actions.view(-1, 1)).squeeze()
        distr = Categorical(logits=act_logits)

        # advantage
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # policy loss
        log_ratio = logits - old_logits
        ratio = torch.exp(log_ratio)
        pg_losses = - advs * ratio
        pg_losses2 = - advs * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))

        # entropy term
        entropy = torch.mean(distr.entropy())

        # value loss
        vpredclipped = old_vals + torch.clamp(vals - old_vals, -cliprange, cliprange)
        vf_losses1 = (vals - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))

        # some useful stats
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > cliprange).type(torch.FloatTensor))
        approxkl = .5 * torch.mean(log_ratio ** 2)

        self.optimizer.zero_grad()
        loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef
        loss.backward()
        self.optimizer.step()

        return pg_loss, entropy, vf_loss, clipfrac, approxkl


def test():
    from gym.spaces import Box, Discrete
    import numpy as np

    obs_space = Box(low=0, high=255, shape=(84, 84, 2), dtype=np.uint8)
    a_space = Discrete(10)

    policy = CNNPolicy(obs_space, a_space, 1, 1, 1)

    obs = torch.tensor(np.random.rand(10, 2, 84, 84).astype(np.float32))
    returns = torch.FloatTensor(np.random.rand(10))
    actions, logits, vals = policy.step(obs)

    policy.train(1, obs, returns, actions, vals, logits)


if __name__ == '__main__':
    test()
