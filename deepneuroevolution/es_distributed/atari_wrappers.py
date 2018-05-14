import numpy as np
from collections import deque
from PIL import Image
import gym
from gym import spaces
import cv2
from gym.spaces.box import Box
from retro_contest.local import make
import random

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip
        self.viewer = None

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(np.max(np.stack(self._obs_buffer), axis=0))
            return np.max(np.stack(self._obs_buffer), axis=0)
        else:
            return np.max(np.stack(self._obs_buffer), axis=0)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, show_warped=False):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))
        self.viewer = None
        self.show_warped = show_warped

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
            resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1))

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human' and self.show_warped:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = self._observation(self.env._render('rgb_array', close)) * np.ones([1, 1, 3], dtype=np.uint8)
            self.viewer.imshow(img)
            return img
        else:
            return self.env._render(mode, close)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class ScaledFloatFrame(gym.ObservationWrapper):
    def _observation(self, obs):
    # careful! This undoes the memory optimization, use
    # with smaller replay buffers only.
        return np.array(obs).astype(np.float32) / 255.0

class DiscretizeActions(gym.Wrapper):
    def __init__(self, env):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.temp_action = env.action_space
        self.action_space = spaces.Discrete(5 ** int(np.prod(env.action_space.shape)))

    def _step(self, action):
        cont_action = self.temp_action.low.copy()
        for i in range(cont_action.size):
            cont_action[i] += (self.temp_action.high[i] - self.temp_action.low[i]) * float(int(action) % 5) / 4.0
            action = int(action / 5)
        return self.env.step(cont_action)

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
        self.real_reward = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self.real_reward = rew
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info



class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, gray=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.gray = gray
        n_ch = 1 if gray else 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, n_ch), dtype=np.uint8)

    def observation(self, frame):
        if self.gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA).\
            reshape(self.height, self.width, -1)
        return frame

class RandomEnvironment(gym.Wrapper):
    """
    Randomly choose level and state after reset is called.
    Warning: this environment should be the first env Wrapper!
    """
    def __init__(self, env, game_states):
        super(RandomEnvironment, self).__init__(env)
        self.game_states = game_states

    def reset(self, **kwargs):
        self.env.close()
        game, state = random.choice(self.game_states)
        self.env = make(game, state)
        return self.env.reset(**kwargs)

# def wrap_deepmind(env, episode_life=True, clip_rewards=True):
# def wrap_deepmind(env, episode_life=False, skip=4, stack_frames=4, noop_max=30, noops=None, show_warped=False):
#     """Configure environment for DeepMind-style Atari.
#
#     Note: this does not include frame stacking!"""
#     if episode_life:
#         env = EpisodicLifeEnv(env)
#     env = NoopResetEnv(env, noop_max=noop_max)
#     if noops:
#         env.override_num_noops = noops
#     if skip > 1:
#         assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
#         env = MaxAndSkipEnv(env, skip=4)
#     if 'FIRE' in env.unwrapped.get_action_meanings():
#         env = FireResetEnv(env)
#     env = WarpFrame(env, show_warped=show_warped)
#     if stack_frames > 1:
#         env = FrameStack(env, stack_frames)
#     env = ScaledFloatFrame(env)
#     return env

def _process_frame84(frame):
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame

class SonicRescale84x84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(SonicRescale84x84, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def observation(self, observation):
        return _process_frame84(observation)

def wrap_deepmind(env_id, state_id, game_states=None):
    env = make(game=env_id, state=state_id)
    env = RandomEnvironment(env, game_states)
    env = SonicDiscretizer(env)
    env = SonicRescale84x84(env)
    env = AllowBacktracking(env)
    env = FrameStack(env, 4)
    return env