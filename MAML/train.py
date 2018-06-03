import sonic_utils
import utils
from model import CNNPolicy
import numpy as np
from time import time
from baselines import logger
from collections import deque
import pandas as pd


def traj_segment_generator(model, env, horizon, sample):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
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

    while True:
        ac, ac_logit, vpred = model.step(ob, sample)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {
                "ob": obs, "rew": rews,
                "vpred": vpreds, "new": news,
                "ac": acs, "nextvpred": float(vpred) * (1 - new),
                "ac_logits": ac_logits, "ep_infos": ep_infos,
            }
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_infos = []

        i = t % horizon
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


def add_vtarg(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = gaelam + seg["vpred"]


def train(config, exp_name='test'):

    train_params = config['train_params']
    env_params = config['env_params']
    log_params = config["log"]

    savedir = None
    if log_params["log_dir"] is not None:
        savedir = utils.prepare_exp_dir(config, exp_name)

    env = sonic_utils.make_from_config(env_params)

    model = CNNPolicy(
        env.observation_space, env.action_space, train_params["vf_coef"],
        train_params["ent_coef"], train_params["lr"], train_params["max_grad_norm"]
    )

    if train_params["weights"] is not None:
        model.load(train_params["weights"], train_params["load_adam_params"])

    seg_gen = traj_segment_generator(
        model, env, train_params['n_steps'], sample=True)

    total_steps = 0
    updates = 0
    t0 = time()
    epinfobuf = deque(maxlen=train_params["ep_info_len"])
    seg_inds = np.arange(train_params['n_steps'])
    n_batches = train_params["n_steps"] // train_params["batch_size"]
    loss_vals = []
    while True:
        if total_steps > train_params['max_steps']:
            break

        # get batch
        seg = seg_gen.__next__()
        add_vtarg(seg, train_params['gamma'], train_params['lam'])

        # add episode info
        epinfobuf.extend(seg['ep_infos'])

        for _ in range(train_params["n_opt_epochs"]):
            np.random.shuffle(seg_inds)
            for i in range(n_batches):
                start = i * train_params["batch_size"]
                end = (i + 1) * train_params["batch_size"]
                inds = seg_inds[start:end]

                losses = model.train(
                    train_params['cliprange'], seg['ob'][inds],
                    seg['tdlamret'][inds], seg['ac'][inds],
                    seg['vpred'][inds], seg["ac_logits"][inds]
                )
                loss_vals.append([l.detach().numpy() for l in losses])

        total_steps += train_params['n_steps']
        updates += 1

        if log_params["log"] and (updates % log_params["log_interval"] == 0 or updates == 1):

            tnow = time()
            fps = int(total_steps / (tnow - t0))
            # ev = explained_variance(values, returns)
            logger.logkv("total_steps", total_steps)
            logger.logkv("nupdates", updates)
            logger.logkv("fps", fps)
            logger.logkv('eprewmean', np.mean([epinfo['r'] for epinfo in epinfobuf if 'r' in epinfo]))
            logger.logkv('eprewmean_exp', np.mean([epinfo['r_exp'] for epinfo in epinfobuf if 'r_exp' in epinfo]))
            logger.logkv('eplenmean', np.mean([epinfo['l'] for epinfo in epinfobuf if 'l' in epinfo]))
            logger.logkv('time_elapsed', tnow - t0)

            for loss_val, loss_name in zip(np.mean(loss_vals, axis=0), model.loss_names):
                logger.logkv(loss_name, loss_val)
            logger.dumpkvs()

            del loss_vals[:]

        # save last weights
        if log_params['save_last'] and savedir is not None:
            fpath = savedir / 'last.pt'
            model.save(fpath)

        # save on save period
        if (updates % log_params["save_interval"] == 0 or updates == 1) and savedir is not None:
            fpath = savedir / '{}.pt'.format(updates)
            model.save(fpath)

    return epinfobuf


def test():
    args = utils.get_args()

    config = utils.load_config(args.config)

    if "game_states" in config['env_params']:
        game_states = pd.read_csv(config['env_params']["game_states"]).values.tolist()

        all_means = []
        for game, state in game_states:
            config['env_params']["game_states"] = [(game, state)]
            epinfobuf = train(config, args.exp_name)

            rewards = [epinfo['r'] for epinfo in epinfobuf if 'r' in epinfo]
            print("{} {} {:.2f} {:.2f}".format(game, state, np.max(rewards), np.mean(rewards)))
            all_means.append(np.mean(rewards))

        print("final result {:.2f}".format(np.mean(all_means)))
    else:
        train(config, args.exp_name)


if __name__ == '__main__':
    test()
