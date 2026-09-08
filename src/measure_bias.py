"""Reproduce the Table 4 action-dependent-normalization-bias measurement
(paper Appendix F.1 / app:action_bias, using the exact formulas of Appendix B / app:bias):

  Fix a state s. Sample K actions a_i ~ pi_theta(.|s). Let
    g_i = grad_theta log pi_theta(a_i|s)
    A_i = r_i + gamma * V(s'_i) * (1 - terminated_i) - V(s)      (TD-error / advantage)
  Then:
    v_unbiased(s)    = mean_i[ A_i * g_i ]                        ("unbiased update", alpha=1)
    v_intentional(s) = mean_i[ (A_i / ||g_i||^2) * g_i ]          (actual Intentional-PG normalization)
    cos_sim(s)       = cosine(v_unbiased(s), v_intentional(s))

This isolates exactly the mechanism in Appendix B (eq. pg_bias_expect_std vs
pg_bias_expect_norm): no RMSProp entrywise scaling, no eligibility trace, no
adaptive delta clipping/normalization, and no entropy term -- Appendix F.1
measures this at lambda=0, which collapses the eligibility trace to g_i exactly,
and the paper's own math for this experiment (Appendix B) never introduces
those other Algorithm-3 mechanisms. See notes/step1_report.md for the full
justification of this reading of the protocol.

States are sampled from an on-policy rollout of the frozen (post-training)
policy, using the checkpoint's frozen observation/reward normalization
statistics (not further updated), exactly as in standard evaluation-mode
handling of running-normalization wrappers.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.distributions import Normal

from common import Actor, Critic, frozen_normalize_obs, frozen_scale_reward, flat_grad
import gymnasium as gym


def load_checkpoint(path):
    ckpt = torch.load(path, weights_only=False)
    actor = Actor(n_obs=ckpt["n_obs"], n_actions=ckpt["n_actions"], hidden_size=ckpt["hidden_size"])
    actor.load_state_dict(ckpt["policy_state_dict"])
    critic = Critic(n_obs=ckpt["n_obs"], hidden_size=ckpt["hidden_size"])
    critic.load_state_dict(ckpt["value_state_dict"])
    return ckpt, actor, critic


def collect_states(env_name, actor, obs_mean, obs_var, obs_eps, seed, n_states):
    env = gym.make(env_name)
    raw = env.unwrapped
    obs, _ = env.reset(seed=seed)
    states = []
    while len(states) < n_states:
        obs_norm = frozen_normalize_obs(obs, obs_mean, obs_var, obs_eps)
        with torch.no_grad():
            mu, std = actor(torch.from_numpy(obs_norm).float())
            a = Normal(mu, std).sample().numpy()
        a_clipped = np.clip(a, env.action_space.low, env.action_space.high)
        states.append((raw.data.qpos.copy(), raw.data.qvel.copy()))
        obs, r, terminated, truncated, info = env.step(a_clipped)
        if terminated or truncated:
            obs, _ = env.reset()
    return states, env


def measure_state_bias(raw_env, actor, critic, obs_mean, obs_var, obs_eps,
                        reward_var, reward_eps, gamma, qpos, qvel, n_actions,
                        act_low, act_high, rng):
    raw_env.set_state(qpos, qvel)
    obs = raw_env._get_obs()
    obs_norm = frozen_normalize_obs(obs, obs_mean, obs_var, obs_eps)
    obs_t = torch.from_numpy(obs_norm).float()

    with torch.no_grad():
        mu, std = actor(obs_t)
        v_s = critic(obs_t).item()
    dist_fixed = Normal(mu, std)
    actions = dist_fixed.sample((n_actions,))

    n_params = sum(p.numel() for p in actor.parameters())
    unbiased_sum = torch.zeros(n_params)
    intentional_sum = torch.zeros(n_params)
    A_list = []
    gnormsq_list = []

    for i in range(n_actions):
        a_i = actions[i]
        a_i_np = a_i.detach().numpy()
        a_clipped = np.clip(a_i_np, act_low, act_high)

        raw_env.set_state(qpos, qvel)
        obs_prime, r, terminated, truncated, info = raw_env.step(a_clipped)
        obs_prime_norm = frozen_normalize_obs(obs_prime, obs_mean, obs_var, obs_eps)
        r_scaled = frozen_scale_reward(r, reward_var, reward_eps)
        with torch.no_grad():
            v_sprime = critic(torch.from_numpy(obs_prime_norm).float()).item()
        termination_mask = 0.0 if terminated else 1.0
        A_i = float(r_scaled + gamma * v_sprime * termination_mask - v_s)

        actor.zero_grad(set_to_none=True)
        mu2, std2 = actor(obs_t)
        log_prob = Normal(mu2, std2).log_prob(a_i).sum()
        log_prob.backward()
        g_i = flat_grad(actor.parameters())
        g_norm_sq = torch.dot(g_i, g_i).item()

        unbiased_sum += A_i * g_i
        intentional_sum += (A_i / max(g_norm_sq, 1e-12)) * g_i
        A_list.append(A_i)
        gnormsq_list.append(g_norm_sq)

    v_unbiased = (unbiased_sum / n_actions)
    v_intentional = (intentional_sum / n_actions)
    denom = v_unbiased.norm().item() * v_intentional.norm().item()
    cos_sim = torch.dot(v_unbiased, v_intentional).item() / denom if denom > 1e-20 else float("nan")

    diag = {
        "mean_A": float(np.mean(A_list)),
        "mean_g_norm_sq": float(np.mean(gnormsq_list)),
        "v_unbiased_norm": v_unbiased.norm().item(),
        "v_intentional_norm": v_intentional.norm().item(),
    }
    return cos_sim, diag


def run(checkpoint_path, out_raw_csv, n_states=1000, n_actions=1000, state_seed=None, debug=False):
    ckpt, actor, critic = load_checkpoint(checkpoint_path)
    env_name = ckpt["env_name"]
    seed = ckpt["seed"]
    gamma = ckpt["gamma"]
    obs_mean, obs_var, obs_eps = ckpt["obs_mean"], ckpt["obs_var"], ckpt["obs_epsilon"]
    reward_var, reward_eps = ckpt["reward_var"], ckpt["reward_epsilon"]

    if state_seed is None:
        state_seed = seed + 1_000_000  # distinct from training seed, deterministic per (env,seed)

    torch.manual_seed(state_seed)
    np.random.seed(state_seed)

    states, env = collect_states(env_name, actor, obs_mean, obs_var, obs_eps, state_seed, n_states)
    raw_env = env.unwrapped
    act_low, act_high = env.action_space.low, env.action_space.high

    rng = np.random.default_rng(state_seed)
    os.makedirs(os.path.dirname(out_raw_csv), exist_ok=True)
    t0 = time.time()
    with open(out_raw_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state_idx", "cos_sim", "mean_A", "mean_g_norm_sq", "v_unbiased_norm", "v_intentional_norm"])
        for idx, (qpos, qvel) in enumerate(states):
            cos_sim, diag = measure_state_bias(
                raw_env, actor, critic, obs_mean, obs_var, obs_eps,
                reward_var, reward_eps, gamma, qpos, qvel, n_actions,
                act_low, act_high, rng,
            )
            writer.writerow([idx, cos_sim, diag["mean_A"], diag["mean_g_norm_sq"],
                              diag["v_unbiased_norm"], diag["v_intentional_norm"]])
            if debug and (idx + 1) % max(1, n_states // 10) == 0:
                elapsed = time.time() - t0
                print(f"[{env_name} seed={seed}] state {idx+1}/{n_states} cos_sim={cos_sim:.4f} elapsed={elapsed:.1f}s")
    env.close()
    return out_raw_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_raw_csv", type=str, required=True)
    parser.add_argument("--n_states", type=int, default=1000)
    parser.add_argument("--n_actions", type=int, default=1000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    run(args.checkpoint, args.out_raw_csv, n_states=args.n_states, n_actions=args.n_actions, debug=args.debug)
