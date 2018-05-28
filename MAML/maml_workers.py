import numpy as np
import Pyro4
import sonic_utils
from model import CNNPolicy
from train import add_vtarg_and_adv


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

    def initialize(self, config, weights):
        self.config = config
        train_params = config['train_params']
        env_params = config['env_params']

        self.env = sonic_utils.make_from_config(env_params, True)
        self.model = CNNPolicy(
            self.env.observation_space, self.env.action_space, train_params["vf_coef"],
            train_params["ent_coef"], train_params["lr"], train_params["max_grad_norm"]
        )
        self.set_model_weights(weights)

    def set_model_weights(self, weights):
        self.model.load_params(weights)

    @property
    def initialized(self):
        return self.model is not None and self.env is not None and self.config is not None

    def _train(self):

        ob = None
        new = True
        train_params = self.config['train_params']
        seg_inds = np.arange(train_params['n_steps'])
        n_batches = train_params["n_steps"] // train_params["batch_size"]
        loss_vals = []
        for i in range(train_params["train_traj"]):

            # sample trajectory
            traj = sample_trajectory(self.model, self.env, train_params["n_steps"], True, ob, new)
            add_vtarg_and_adv(traj, train_params['gamma'],  train_params['lam'])

            ob = traj["last_ob"]
            new = traj["last_new"]

            # run training
            for _ in range(train_params["n_opt_epochs"]):
                np.random.shuffle(seg_inds)
                for i in range(n_batches):
                    start = i * train_params["batch_size"]
                    end = (i + 1) * train_params["batch_size"]
                    inds = seg_inds[start:end]

                    losses = self.model.train(
                        train_params['cliprange'], traj['ob'][inds],
                        traj["adv"][inds], traj['tdlamret'][inds], traj['ac'][inds],
                        traj['vpred'][inds], traj["ac_logits"][inds]
                    )
                    loss_vals.append([l.detach().numpy() for l in losses])

        return loss_vals, ob, new

    def run(self, params):
        # set model params
        self.model.load_params(params)

        # first choose environment
        self.env.sample()

        # then train model
        loss_vals, ob, new = self._train()

        # then collect data for metalerning
        n_steps = self.config["train_params"]["meta_n_steps"]
        traj = sample_trajectory(self.model, self.env, n_steps, True, ob, new)

        return traj