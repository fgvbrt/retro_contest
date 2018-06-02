import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import utils
import numpy as np



class NatureCNN(nn.Module):
    def __init__(self, n_ac, ob_space):
        super(NatureCNN, self).__init__()
        nc, nh, nw = ob_space.shape
        out_h = utils.convs_out_dim(nh, [8, 4, 3], [0, 0, 0], [4, 2, 1])
        out_w = utils.convs_out_dim(nw, [8, 4, 3], [0, 0, 0], [4, 2, 1])
        self.inp_shape = (1, nc, nh, nw)
        self.conv_out = out_w * out_h * 64

        self.conv1 = nn.Conv2d(nc, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.conv_out, 512)
        self.action_head = nn.Linear(512, n_ac)
        self.value_head = nn.Linear(512, 1)

        # initialize as in openai baselines
        self._init()

    def _init(self):
        for m in (self.conv1, self.conv2, self.conv3, self.fc1):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.action_head.weight, 0.01)
        nn.init.constant_(self.action_head.bias, 0.0)

        nn.init.orthogonal_(self.value_head.weight, 1)
        nn.init.constant_(self.value_head.bias, 0.0)

        # init gradients
        tmp = torch.from_numpy(np.random.rand(*self.inp_shape).astype(np.float32))
        tmp = self(tmp)
        tmp_loss = tmp[0].mean() + tmp[1].mean()
        tmp_loss.backward()
        self.zero_grad()

    def get_grads(self):
        return [p.grad for p in self.parameters() if p.grad is not None]

    def add_grads(self, grads):
        # TODO: надо бы проверить, что верно проинициализировано
        i = 0
        for p in self.parameters():
            if p.grad is not None:
                p.grad.add_(grads[i])
                i += 1
        # check that all grads were processed
        assert i == len(grads)

    def forward(self, inp):
        x = F.relu(self.conv1(inp))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.conv_out)
        x = F.relu(self.fc1(x))

        act_logits = F.log_softmax(self.action_head(x), -1)
        vals = self.value_head(x).squeeze(dim=1)

        return act_logits, vals


class CNNPolicy(object):
    def __init__(self, ob_space, ac_space, vf_coef, ent_coef, lr, max_grad_norm):
        self.model = NatureCNN(ac_space.n, ob_space)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.loss_names = ("pg_loss", "entropy", "vf_loss", "clipfrac", "approxkl")
        self.init_lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr)

    def value(self, obs):
        act_logits, vals = self.model(obs)
        return vals

    def get_grads(self):
        return self.model.get_grads()

    def add_grads(self, grads):
        self.model.add_grads(grads)

    def zero_grad(self):
        self.model.zero_grad()

    def get_weights(self):
        return self.model.state_dict()

    def load_weights(self, weights):
        return self.model.load_state_dict(weights)

    def save(self, fname):
        state_dict = {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
        }
        torch.save(state_dict, str(fname))

    def load(self, fname, restore_opt=None):
        state_dict = torch.load(str(fname))
        self.model.load_state_dict(state_dict["model"])

        if restore_opt == "all":
            # keep learning rate
            for gr in state_dict['opt']["param_groups"]:
                if 'lr' in gr:
                    gr['lr'] = self.init_lr
            self.optimizer.load_state_dict(state_dict["opt"])
        elif restore_opt == "weight_stats":
            for gr in state_dict['opt']["param_groups"]:
                # first learning rate
                if 'lr' in gr:
                    gr['lr'] = self.init_lr

                # then zero step and leave only weight stats
                for p in gr["params"]:
                    state = state_dict['opt']['state']
                    if p in state and 'step' in state[p]:
                        state[p]['step'] = 0

            self.optimizer.load_state_dict(state_dict["opt"])
        elif restore_opt is None or restore_opt is False:
            pass
        else:
            raise ValueError('unknown option for restore opt'.format(restore_opt))

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

    def _backprop(self, cliprange, obs, returns, actions, old_vals, old_logits):
        # TODO: may store all of it in tensors already
        # TODO: calculate advantages for memory efficiency reasons
        obs = torch.from_numpy(obs)
        actions = torch.from_numpy(actions)
        old_vals = torch.from_numpy(old_vals)
        old_logits = torch.from_numpy(old_logits)
        returns = torch.from_numpy(returns)

        # current logits and vals
        act_logits, vals = self.model(obs)
        logits = act_logits.gather(1, actions.view(-1, 1)).squeeze()
        distr = Categorical(logits=act_logits)

        # advantage
        advs = returns - old_vals
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

        return pg_loss, entropy, vf_loss, clipfrac, approxkl

    def train(self, cliprange, obs, returns, actions, old_vals, old_logits):
        pg_loss, entropy, vf_loss, clipfrac, approxkl = \
            self._backprop(cliprange, obs, returns, actions, old_vals, old_logits)

        self.optimizer.zero_grad()
        loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return pg_loss, entropy, vf_loss, clipfrac, approxkl

    def accumulate_grad(self, cliprange, obs, returns, actions, old_vals, old_logits):
        pg_loss, entropy, vf_loss, clipfrac, approxkl = \
            self._backprop(cliprange, obs, returns, actions, old_vals, old_logits)

        loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef
        loss.backward()

        return pg_loss, entropy, vf_loss, clipfrac, approxkl
