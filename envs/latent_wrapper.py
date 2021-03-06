# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import gym
from gym import Wrapper

class LatentWrapper(Wrapper):

  def __init__(
      self,
      env,
      # latent type and dimension
      num_latents=None,
      latent_type='discrete_uniform',
      # execute an episode with the same predefined latent, does not resample
      preset_latent=None,
      # resample latents within episode
      min_steps_before_resample=10,
      resample_prob=0.):

    super(LatentWrapper, self).__init__(env)
    self._latent_type = latent_type
    if num_latents is None:
      self._num_latents = 0
    else:
      self._num_latents = num_latents
    self._preset_latent = preset_latent

    # attributes for controlling latent resampling
    self._min_steps_before_resample = min_steps_before_resample
    self._resample_prob = resample_prob

    if isinstance(self.env.observation_space, gym.spaces.Dict):
      size = self.env.observation_space.spaces['observation'].shape[0] + self._num_latents
    else:
      size = self.env.observation_space.shape[0] + self._num_latents
    self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype='float32')

  def _remake_time_step(self, cur_obs):
    if isinstance(self.env.observation_space, gym.spaces.Dict):
      cur_obs = cur_obs['observation']

    if self._num_latents == 0:
      return cur_obs
    else:
      return np.concatenate([cur_obs, self.latent])

  def _set_latent(self):
    if self._num_latents:
      if self._preset_latent is not None:
        self.latent = self._preset_latent
        print('Latent:', self.latent)
      elif self._latent_type == 'discrete_uniform':
        self.latent = np.random.multinomial(
            1, [1. / self._num_latents] * self._num_latents)
      elif self._latent_type == 'gaussian':
        self.latent = np.random.multivariate_normal(
            np.zeros(self._num_latents), np.eye(self._num_latents))
      elif self._latent_type == 'cont_uniform':
        self.latent = np.random.uniform(
            low=-1.0, high=1.0, size=self._num_latents)

  def reset(self):
    cur_obs = self.env.reset()
    self._set_latent()
    self._step_count = 0
    return self._remake_time_step(cur_obs)

  def step(self, action):
    cur_obs, reward, done, info = self.env.step(action)
    self._step_count += 1
    if self._preset_latent is None and self._step_count >= self._min_steps_before_resample and np.random.random(
    ) < self._resample_prob:
      self._set_latent()
      self._step_count = 0
    return self._remake_time_step(cur_obs), reward, done, info

  def close(self):
    return self.env.close()
