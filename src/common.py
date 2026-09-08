import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASELINE_DIR = os.path.join(os.path.dirname(_THIS_DIR), "baseline")
if _BASELINE_DIR not in sys.path:
    sys.path.insert(0, _BASELINE_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Normal

from intentional_ac import Actor, Critic, IntentionalAC  # noqa: E402
from normalization_wrappers import NormalizeObservation, ScaleReward, SampleMeanStd  # noqa: E402


def make_training_env(env_name, gamma):
    env = gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    return env


def frozen_normalize_obs(raw_obs, obs_mean, obs_var, epsilon=1e-8):
    # NormalizeObservation's SampleMeanStd picks up an extra leading dim of size 1
    # (it always updates with np.array([obs])), so reshape back to raw_obs's shape.
    mean = np.asarray(obs_mean).reshape(raw_obs.shape)
    var = np.asarray(obs_var).reshape(raw_obs.shape)
    return (raw_obs - mean) / np.sqrt(var + epsilon)


def frozen_scale_reward(raw_reward, reward_var, epsilon=1e-8):
    var = float(np.asarray(reward_var).reshape(-1)[0])
    return float(raw_reward) / np.sqrt(var + epsilon)


def flat_grad(params):
    """Flatten .grad of an iterable of parameters into a single 1D torch tensor (zeros where grad is None)."""
    parts = []
    for p in params:
        if p.grad is None:
            parts.append(torch.zeros(p.numel()))
        else:
            parts.append(p.grad.reshape(-1).clone())
    return torch.cat(parts)
