import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import Pyro4
import sonic_utils
from model import CNNPolicy
from train import add_vtarg
import argparse
import pickle
import utils
from baselines import logger
from time import time
from collections import deque
import torch
import pickle

torch.set_num_threads(1)
logger.set_level(logger.DEBUG)


def sample_trajectory(model, env, horizon, sample, ob=None, new=False):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype

    if ob is None:
        new = True  # marks if we're on first timestep of an episode
        ob = env.reset()

    ep_infos = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    ac_logits = np.zeros(horizon, 'float32')
    vpred = 0

    for i in range(horizon):
        ac, ac_logit, vpred = model.step(ob, sample)

        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        ac_logits[i] = ac_logit

        ob, rew, new, info = env.step(ac)
        rews[i] = rew

        if new:
            # game_name = env.unwrapped.game_name
            # state_name = env.unwrapped.state_name
            if "episode" in info:
                ep_infos.append(info["episode"])

            ob = env.reset()

        t += 1

    return {
        "ob": obs, "rew": rews,
        "vpred": vpreds, "new": news,
        "ac": acs, "nextvpred": float(vpred) * (1 - new),
        "ac_logits": ac_logits, "ep_infos": ep_infos,
        "last_ob": ob, "last_new": new
    }


@Pyro4.expose
class MAMLWorker(object):

    def __init__(self):
        self.model = None
        self.train_params = None
        self.env = None
        self.config = None
        self.meta_steps = None
        self.updates = 0
        self.start_t = None
        self.epinfobuf = deque(maxlen=100)

    def initialize(self, config, weights):
        self.config = config
        train_params = config['train_params']
        env_params = config['env_params']

        if self.env is not None:
            self.env.close()

        self.env = sonic_utils.make_from_config(env_params, True)

        self.model = CNNPolicy(
            self.env.observation_space, self.env.action_space, train_params["vf_coef"],
            train_params["ent_coef"], train_params["lr"], train_params["max_grad_norm"]
        )

        self.set_model_weights(weights)
        self.start_t = time()
        logger.debug("worker initialized")

    def set_model_weights(self, weights):
        weights = utils.unpickle(weights)

        self.model.load_weights(weights)

    @property
    def initialized(self):
        return self.model is not None and self.env is not None and self.config is not None

    def _train(self, n_traj, train=True, ob=None, new=True):
        train_params = self.config['train_params']

        if train:
            train_fn = self.model.train
            n_opt_epochs = train_params["n_opt_epochs"]
        else:
            train_fn = self.model.accumulate_grad
            n_opt_epochs = 1

        seg_inds = np.arange(train_params['n_steps'])
        n_batches = train_params["n_steps"] // train_params["batch_size"]
        loss_vals = []
        for i in range(n_traj):

            # sample trajectory
            traj = sample_trajectory(self.model, self.env, train_params["n_steps"], True, ob, new)
            add_vtarg(traj, train_params['gamma'], train_params['lam'])

            ob = traj["last_ob"]
            new = traj["last_new"]

            if not train or train_params["meta_algo"] == "reptile":
                self.epinfobuf.extend(traj['ep_infos'])

            # run training
            for _ in range(n_opt_epochs):
                np.random.shuffle(seg_inds)
                for i in range(n_batches):
                    start = i * train_params["batch_size"]
                    end = (i + 1) * train_params["batch_size"]
                    inds = seg_inds[start:end]

                    losses = train_fn(
                        train_params['cliprange'], traj['ob'][inds],
                        traj['tdlamret'][inds], traj['ac'][inds],
                        traj['vpred'][inds], traj["ac_logits"][inds]
                    )
                    loss_vals.append([l.detach().numpy() for l in losses])

        return loss_vals, ob, new

    def run(self, weights=None):
        logger.debug("start worker run")

        # set model params
        if weights is not None:
            self.set_model_weights(weights)

        # copy initial weights
        # TODO: optimize with copy remove ugly weights receiving
        weights = pickle.dumps([p for p in self.model.model.parameters() if p.grad is not None])
        weights = pickle.loads(weights)

        # first sample environment
        self.env.sample()

        # then train model
        train_params = self.config["train_params"]
        loss_vals, ob, new = self._train(train_params["n_traj"], True)
        logger.debug("worker training finished")

        # collect samples for gradient only for meta learning algo
        k = 0
        if train_params["meta_algo"] == "maml":
            # then collect accumulate gradients for metalearning
            loss_vals, _, _ = self._train(train_params["n_traj2"], False, ob, new)
            logger.debug("worker gradients accumulation finished")
            meta_grads = self.model.get_grads()
        elif train_params["meta_algo"] == "reptile":
            k = 1
            cur_weights = [p for p in self.model.model.parameters() if p.grad is not None]
            meta_grads = [w_old - w_new for w_old, w_new in zip(weights, cur_weights)]
        else:
            raise ValueError("unknown meta algo {}".format(train_params["meta_algo"]))

        res = {
            "game_name": self.env.unwrapped.gamename,
            "state_name": self.env.unwrapped.statename,
            "grads": meta_grads
        }

        self.updates += 1
        total_steps = self.updates * train_params["n_steps"] * (train_params["n_traj2"] * k + train_params["n_traj"])
        if self.updates % self.config["log"]["log_interval"] == 0 or self.updates == 1:
            epinfobuf = self.epinfobuf

            tnow = time()
            fps = int(total_steps / (tnow - self.start_t))
            # ev = explained_variance(values, returns)
            logger.logkv("total_steps", total_steps)
            logger.logkv("nupdates", self.updates)
            logger.logkv("fps", fps)
            logger.logkv('eprewmean', np.mean([epinfo['r'] for epinfo in epinfobuf if 'r' in epinfo]))
            logger.logkv('eprewmean_exp', np.mean([epinfo['r_exp'] for epinfo in epinfobuf if 'r_exp' in epinfo]))
            logger.logkv('eplenmean', np.mean([epinfo['l'] for epinfo in epinfobuf if 'l' in epinfo]))
            logger.logkv('time_elapsed', tnow - self.start_t)

            for loss_val, loss_name in zip(np.mean(loss_vals, axis=0), self.model.loss_names):
                logger.logkv(loss_name, loss_val)
            logger.dumpkvs()

        return pickle.dumps(res)


def main():
    def _get_args():
        parser = argparse.ArgumentParser(description="Run commands")
        parser.add_argument('--name', type=str, default='1', help="Name of server")
        parser.add_argument('--host', type=str, default='localhost', help="Host name.")
        parser.add_argument('--port', type=int, default=9091, help="Host port.")
        parser.add_argument('--nathost', type=str, default=None, help="Nat name.")
        parser.add_argument('--natport', type=int, default=None, help="Nat port.")
        parser.add_argument('--ns_host', type=str, default='94.45.222.176', help="Name server host.")
        return parser.parse_args()

    args = _get_args()
    worker = MAMLWorker()

    # for example purposes we will access the daemon and name server ourselves and not use serveSimple
    with Pyro4.Daemon(host=args.host, port=args.port, nathost=args.nathost, natport=args.natport) as daemon:
        uri = daemon.register(worker)
        name = "worker.{}.port.{}".format(args.name, args.port)

        with Pyro4.locateNS(host=args.ns_host) as ns:
            ns.register(name, uri)

        print("Sampler ready: name {} uri {}".format(name, uri))
        daemon.requestLoop()


if __name__ == '__main__':
    main()
