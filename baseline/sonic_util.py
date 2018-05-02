"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np

from baselines.common.atari_wrappers import WarpFrame, FrameStack
import gym_remote.client as grc
import retro
import retro_contest
import random


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


def make_remote_env(stack=True, scale_rew=True, socket_dir='/tmp'):
    """
    Create an environment with some standard wrappers.
    """
    env = grc.RemoteEnv(socket_dir)
    env = SonicDiscretizer(env)
    env = AllowBacktracking(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    env = EpisodeInfo(env)
    return env


def make_env(game, state, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = make(game, state)
    env = retro_contest.StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=4500)

    env = SonicDiscretizer(env)
    env = AllowBacktracking(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    env = EpisodeInfo(env)
    return env


def make_rand_env(game_states, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    game, state = game_states[0]
    env = make(game, state)

    env = RandomEnvironmen(env, game_states)
    env = retro_contest.StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=4500)

    env = SonicDiscretizer(env)
    env = AllowBacktracking(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    env = EpisodeInfo(env)
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

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01


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

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
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
            elif isinstance(info, dict):
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

    def reset(self, **kwargs):
        self.env.close()
        game, state = random.choice(self.game_states)
        self.env = make(game, state)
        return self.env.reset(**kwargs)
