"""
Environments and wrappers for Sonic training.
"""
import gym
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from gym import spaces
import cv2
import retro
import gym_remote.client as grc
import retro_contest
from baselines.common.atari_wrappers import FrameStack
from copy import deepcopy
cv2.ocl.setUseOpenCL(False)


def make_from_config(config, maml=False):
    # local training
    config = deepcopy(config)
    if "game_states" in config:

        # file
        if isinstance(config["game_states"], str):
            game_states = pd.read_csv(config["game_states"]).values.tolist()
            config["game_states"] = game_states

        assert isinstance(config["game_states"], list)

        env = make_rand_env(maml=maml, **config)

    # remote testing
    elif "socket_dir" in config:
        env = make_remote_env(**config)
    else:
        raise ValueError("provide game_states or socket dir")

    return env


def make(game, state, discrete_actions=False, bk2dir=None):
    use_restricted_actions = retro.ACTIONS_FILTERED
    if discrete_actions:
        use_restricted_actions = retro.ACTIONS_DISCRETE
    try:
        env = retro.make(game, state, scenario='contest', use_restricted_actions=use_restricted_actions)
    except Exception:
        env = retro.make(game, state, use_restricted_actions=use_restricted_actions)
    if bk2dir:
        env.auto_record(bk2dir)

    return env


def make_remote_env(stack=2, scale_rew=True, color=False,  exp_type='obs', exp_const=0.002,
                    socket_dir='/tmp', small_size=False):
    """
    Create an environment with some standard wrappers.
    """
    env = grc.RemoteEnv(socket_dir)
    env = BackupOriginalData(env)
    env = SonicDiscretizer(env)
    env = AllowBacktracking(env)

    if scale_rew:
        env = RewardScaler(env)

    env = WarpFrame(env, color, small_size)

    if exp_const > 0:
        if exp_type == 'obs':
            env = ObsExplorationReward(env, exp_const, game_specific=False)
        elif exp_type == 'x':
            env = XExplorationReward(env, exp_const, game_specific=False)

    if stack > 1:
        env = FrameStack(env, stack)

    env = EpisodeInfo(env)

    return env


def make_rand_env(game_states, stack=2, scale_rew=True, color=False, exp_type='x',
                  exp_const=0.002, max_episode_steps=4500, maml=False, small_size=False):
    """
    Create an environment with some standard wrappers.
    """
    game, state = game_states[0]
    env = make(game, state)

    if maml:
        env_rand = RandomEnvironmen2(env, game_states)
    else:
        env_rand = RandomEnvironmen(env, game_states)

    env = retro_contest.StochasticFrameSkip(env_rand, n=4, stickprob=0.25)

    env = BackupOriginalData(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    env = SonicDiscretizer(env)
    env = AllowBacktracking(env)

    if scale_rew:
        env = RewardScaler(env)

    env = WarpFrame(env, color, small_size)

    if exp_const > 0:
        if exp_type == 'obs':
            env = ObsExplorationReward(env, exp_const, game_specific=True)
        elif exp_type == 'x':
            env = XExplorationReward(env, exp_const, game_specific=True)

    if stack > 1:
        env = FrameStack(env, stack)

    env = EpisodeInfo(env)

    if maml:
        env.sample = env_rand.sample

    return env


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01


class BackupOriginalData(gym.Wrapper):
    """
    Backup all original information. Should be first wrapper
    """
    def __init__(self, env, reward=True, observation=False):
        super(BackupOriginalData, self).__init__(env)
        self._reward = reward
        self._observation = observation

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self._reward or self._observation:
            info['original'] = {}
            if self._reward:
                info['original']['rew'] = rew
            if self._observation:
                info['original']['obs'] = obs

        return obs, rew, done, info


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


class EpisodeInfo(gym.Wrapper):
    """
    Add information about episode end and total final reward
    """
    def __init__(self, env):
        super(EpisodeInfo, self).__init__(env)
        self._ep_len = 0
        self._ep_rew_total = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._ep_len = 0
        self._ep_rew_total = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._ep_len += 1
        self._ep_rew_total += rew

        if done:
            if "episode" not in info:
                info = {"episode": {"l": self._ep_len, "r": self._ep_rew_total}}
            elif isinstance(info["episode"], dict):
                if "l" not in info["episode"]:
                    info["episode"]["l"] = self._ep_len
                if "r" not in info["episode"]:
                    info["episode"]["r"] = self._ep_rew_total

        return obs, rew, done, info


class RandomEnvironmen(gym.Wrapper):
    """
    Randomly choose level and state after reset is called.
    Warning: this environment should be the first env Wrapper!
    """
    def __init__(self, env, game_states):
        super(RandomEnvironmen, self).__init__(env)
        self.game_states = game_states

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        self.env.close()
        game, state = random.choice(self.game_states)
        self.env = make(game, state)
        return self.env.reset(**kwargs)


class RandomEnvironmen2(gym.Wrapper):
    """
    Randomly choose level and state after reset is called.
    Warning: this environment should be the first env Wrapper!
    """
    def __init__(self, env, game_states):
        super(RandomEnvironmen2, self).__init__(env)
        self.game_states = game_states

    def sample(self):
        game, state = random.choice(self.game_states)
        self.env.close()
        self.env = make(game, state)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class XExplorationReward(gym.Wrapper):
    """
    This should go before any modification of state and after any modification of rewards
    """
    def __init__(self, env, exp_const=0.0002, game_specific=False):
        super(XExplorationReward, self).__init__(env)
        self.exp_const = exp_const
        self._cur_x = 0
        self._exp_reward = 0
        self._ep_rew_total = 0

        env = env.unwrapped

        if game_specific and hasattr(env, "gamename") and hasattr(env, "statename"):
            self.pseudo_counts = defaultdict(lambda: np.zeros(9000, dtype='int'))
            self.game_specific = True
        else:
            self.pseudo_counts = np.zeros(9000, dtype='int')
            self.game_specific = False

    def reset(self, **kwargs):
        self._cur_x = 0
        self._ep_rew_total = 0
        self._exp_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        r_plus = 0
        self._cur_x += info['original']['rew']
        if self.exp_const > 0:

            if self.game_specific:
                env_name = '{}_{}'.format(self.env.unwrapped.gamename, self.env.unwrapped.statename)
                pseudo_counts = self.pseudo_counts[env_name]
            else:
                pseudo_counts = self.pseudo_counts

            i = int(min(self._cur_x, 8999))
            cnt = pseudo_counts[i]
            pseudo_counts[i] += 1

            r_plus = self.exp_const / np.sqrt(cnt + 0.01)

        info['rew_exp'] = r_plus
        self._exp_reward += r_plus
        self._ep_rew_total += rew
        rew += r_plus

        if done:
            if "episode" not in info:
                info["episode"] = {"r": self._ep_rew_total, "r_exp": self._exp_reward}
            elif isinstance(info["episode"], dict):
                info["episode"]["r_exp"] = self._exp_reward
                if "r" not in info["episode"]:
                    info["episode"]["r"] = self._ep_rew_total
            self._ep_rew_total = 0
            self._exp_reward = 0

        return obs, rew, done, info


class ObsExplorationReward(gym.Wrapper):
    """
    This should go before any modification of state and after any modification of rewards
    """
    def __init__(self, env, exp_const=0.0002, game_specific=False):
        super(ObsExplorationReward, self).__init__(env)
        self.exp_const = exp_const

        env = env.unwrapped

        if game_specific and hasattr(env, "gamename") and hasattr(env, "statename"):
            self.pseudo_counts = defaultdict(lambda: np.zeros((42 * 42, 256), dtype='int'))
            self.game_specific = True
        else:
            self.pseudo_counts = np.zeros((42 * 42, 256), dtype='int')
            self.game_specific = False

        self._exp_reward = 0
        self._ep_rew_total = 0

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)

        r_plus = 0
        if self.exp_const > 0:
            # frame = cv2.resize(obs, (84, 84))
            frame = obs
            if frame.shape[2] > 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            s_r = cv2.resize(frame, (42, 42)).ravel()

            if self.game_specific:
                env_name = '{}_{}'.format(self.env.unwrapped.gamename, self.env.unwrapped.statename)
                pseudo_counts = self.pseudo_counts[env_name]
            else:
                pseudo_counts = self.pseudo_counts

            # add observation
            pseudo_counts[np.arange(42 * 42), s_r] += 1

            # calculate current density
            n = pseudo_counts[np.arange(42 * 42), s_r]
            N = pseudo_counts.sum(axis=1)
            p = np.prod(n.astype('float') / N)

            n_after = n + 1
            N_after = N + 1
            p_after = np.prod(n_after.astype('float') / N_after)

            if p_after == p:
                pseudo_cnt = 0
            else:
                pseudo_cnt = p * (1 - p_after) / (p_after - p)
            r_plus = self.exp_const / np.sqrt(pseudo_cnt + 0.01)

        info['rew_exp'] = r_plus
        self._exp_reward += r_plus
        self._ep_rew_total += rew
        rew += r_plus

        if done:
            if "episode" not in info:
                info["episode"] = {"r": self._ep_rew_total, "r_exp": self._exp_reward}
            elif isinstance(info["episode"], dict):
                info["episode"]["r_exp"] = self._exp_reward
                if "r" not in info["episode"]:
                    info["episode"]["r"] = self._ep_rew_total
            self._ep_rew_total = 0
            self._exp_reward = 0

        return obs, rew, done, info

    def reset(self, **kwargs):
        self._ep_rew_total = 0
        self._exp_reward = 0
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, color=False, small_size=False):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.small_size = small_size

        self.width = 84
        self.height = 84

        if small_size:
            self.width = 42
            self.height = 42

        self.color = color
        n_ch = 3 if color else 1
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, n_ch), dtype=np.uint8)

    def observation(self, frame):
        if not self.color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA).\
            reshape(84, 84, -1)

        if self.small_size:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA). \
                reshape(self.width, self.height, -1)

        return frame
