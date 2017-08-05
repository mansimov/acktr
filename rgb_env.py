import numpy as np
from scipy.misc import imresize
from gym.spaces import Discrete, Box, Tuple
from gym import Env
import cv2

class RGBEnv(Env):
    def __init__(self, env, is_rgb=True):
        self._env = env
        self.is_rgb = is_rgb
        if self.is_rgb:
            self._observation_space = Box(low=0.0, high=1.0, shape=(42, 42, 3)) # 42, 42, 3
        else:
            self._observation_space = Box(low=0.0, high=1.0, shape=(42, 42, 1)) # 42, 42, 1
        self.spec = self._env.spec
        self.spec.reward_threshold = self.spec.reward_threshold or float('inf')

    @property
    def action_space(self):
        if isinstance(self._env.action_space, Box):
            ub = np.ones(self._env.action_space.shape)
            return Box(-1 * ub, ub)
        return self._env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    # Taken from universe-starter-agent
    def _process_frame42(self, frame):
        frame = frame[34:34+160, :160]
        # Resize by half, then down to 42x42 (essentially mipmapping). If
        # we resize directly we lose pixels that, when mapped to 42x42,
        # aren't close enough to the pixel boundary.
        frame = cv2.resize(frame, (80, 80)) # 80, 80
        frame = cv2.resize(frame, (42, 42)) # 42, 42
        if self.is_rgb is False:
            frame = frame.mean(2)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        if self.is_rgb is False:
            frame = np.reshape(frame, [42, 42, 1]) # 42, 42, 1
        else:
            frame = np.reshape(frame, [42, 42, 3]) # 42, 42, 3
        return frame

    def reset(self, **kwargs):
        self._env.reset(**kwargs)
        frame = self._process_frame42(self._env.render('rgb_array'))
        return frame

    def step(self, action):
        if isinstance(self._env.action_space, Box):
            # rescale the action
            lb = self._env.action_space.low
            ub = self._env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action

        wrapped_step = self._env.step(scaled_action)
        _, reward, done, info = wrapped_step
        next_frame = self._process_frame42(self._env.render('rgb_array'))

        return next_frame, reward, done, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def __getattr__(self, field):
        """
        proxy everything to underlying env
        """
        if hasattr(self._env, field):
            return getattr(self._env, field)
        raise AttributeError(field)

    def __repr__(self):
        if "object at" not in str(self._env):
            env_name = str(env._env)
        else:
            env_name = self._env.__class__.__name__
        return env_name
