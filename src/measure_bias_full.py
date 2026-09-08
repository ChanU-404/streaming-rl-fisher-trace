"""Table-4-style bias measurement using the *actual* Algorithm 3 / IntentionalOptimizerPolicy
mechanism (RMSProp entrywise preconditioner rho_t=1/v_hat, sigma-bar, and the delta
clip+normalize EMA), rather than the simplified Appendix B illustration used in
measure_bias.py.

For a fixed state s and K sampled actions a_i ~ pi(.|s):
    g_i          = grad_theta log pi(a_i|s)                      (raw actor gradient, no entropy term)
    v_hat        = frozen per-coordinate RMSProp normalizer (checkpoint snapshot, unaffected by lambda)
    g_precond_i  = g_i / v_hat                                   (the actual update DIRECTION, i.e. e/v_hat with lambda=0 -> e=g_i)
    sigma_i      = <g_i, g_i / v_hat> = sum(g_i^2 / v_hat)        (the actual per-sample sigma_tau = <rho g,g>)
    sigma_bc     = frozen bias-corrected sigma-bar from checkpoint (training-time EMA, lambda=0.8)
    alpha_i      = eta_policy / sqrt(sigma_bc * sigma_i)          (the actual Algorithm 3 step size)
    delta_i      = TD-error r_i + gamma*V(s'_i)*(1-terminated_i) - V(s)
    safe_delta_i = frozen adaptive-clip + EMA-normalize of delta_i (checkpoint-frozen clip/norm stats)

    v_unbiased(s)    = mean_i[ safe_delta_i * 1        * g_precond_i ]   ("unbiased update", alpha=1)
    v_intentional(s) = mean_i[ safe_delta_i * alpha_i   * g_precond_i ]  (actual Intentional-PG update)
    cos_sim(s)       = cosine(v_unbiased(s), v_intentional(s))

All frozen quantities (v_hat, sigma-bar, clip/normalize EMA stats) come from the
optimizer's internal state at the end of the 1M-step training run and are held
constant across every (state, action) sample in the measurement -- consistent
with standard eval-mode handling of running statistics, and avoiding
order-dependence across the 1000x1000 branching evaluation.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from torch.distributions import Normal

from common import Actor, Critic, frozen_normalize_obs, frozen_scale_reward, flat_grad
import gymnasium as gym

from measure_bias import collect_states  # reuse state-collection logic verbatim


def load_checkpoint_full(path):
    ckpt = torch.load(path, weights_only=False)
    actor = Actor(n_obs=ckpt["n_obs"], n_actions=ckpt["n_actions"], hidden_size=ckpt["hidden_size"])
    actor.load_state_dict(ckpt["policy_state_dict"])
    critic = Critic(n_obs=ckpt["n_obs"], hidden_size=ckpt["hidden_size"])
    critic.load_state_dict(ckpt["value_state_dict"])

    v_hat_flat = torch.cat([v.reshape(-1) for v in ckpt["policy_opt_rmsprop_v_hat"]])
    n_params = sum(p.numel() for p in actor.parameters())
    assert v_hat_flat.numel() == n_params, f"v_hat size {v_hat_flat.numel()} != actor params {n_params}"

    gamma, lamda = ckpt["gamma"], ckpt["lamda"]
    t_step = ckpt["policy_opt_t_step"]
    sigma_bc = ckpt["policy_opt_sigma"] / (1 - (gamma * lamda) ** t_step)

    clip_t = ckpt["policy_opt_clip_t"]
    norm_t = ckpt["policy_opt_norm_t"]
    frozen = {
        "v_hat_flat": v_hat_flat,
        "sigma_bc": sigma_bc,
        "eta_policy": ckpt["eta_policy"],
        "clip_ema_sq": ckpt["policy_opt_clip_ema_sq"],
        "clip_t": clip_t,
        "delta_abs_ema": ckpt["policy_opt_delta_abs_ema"],
        "norm_t": norm_t,
        "beta_clip": ckpt["policy_opt_beta_clip"],
        "beta_norm": ckpt["policy_opt_beta_norm"],
        "clip_mult": ckpt["policy_opt_clip_mult"],
    }
    return ckpt, actor, critic, frozen


def frozen_process_delta(delta, frozen):
    cap = frozen["clip_mult"] * math.sqrt(frozen["clip_ema_sq"] / (1 - frozen["beta_clip"] ** frozen["clip_t"]))
    clipped = math.copysign(min(abs(delta), cap), delta) if delta != 0 else 0.0
    delta_abs_ema_corrected = frozen["delta_abs_ema"] / (1 - frozen["beta_norm"] ** frozen["norm_t"])
    return clipped / max(delta_abs_ema_corrected, 1e-12)


def measure_state_bias_full(raw_env, actor, critic, obs_mean, obs_var, obs_eps,
                             reward_var, reward_eps, gamma, qpos, qvel, n_actions,
                             act_low, act_high, frozen):
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
    v_hat_flat = frozen["v_hat_flat"]
    diag = {"A": [], "safe_delta": [], "sigma_i": [], "alpha_i": []}

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
        safe_delta_i = frozen_process_delta(A_i, frozen)

        actor.zero_grad(set_to_none=True)
        mu2, std2 = actor(obs_t)
        log_prob = Normal(mu2, std2).log_prob(a_i).sum()
        log_prob.backward()
        g_i = flat_grad(actor.parameters())

        g_precond_i = g_i / v_hat_flat
        sigma_i = torch.dot(g_i, g_precond_i).item()
        alpha_i = frozen["eta_policy"] / math.sqrt(max(frozen["sigma_bc"] * sigma_i, 1e-20))

        unbiased_sum += safe_delta_i * g_precond_i
        intentional_sum += safe_delta_i * alpha_i * g_precond_i

        diag["A"].append(A_i)
        diag["safe_delta"].append(safe_delta_i)
        diag["sigma_i"].append(sigma_i)
        diag["alpha_i"].append(alpha_i)

    v_unbiased = unbiased_sum / n_actions
    v_intentional = intentional_sum / n_actions
    denom = v_unbiased.norm().item() * v_intentional.norm().item()
    cos_sim = torch.dot(v_unbiased, v_intentional).item() / denom if denom > 1e-20 else float("nan")

    out_diag = {
        "mean_A": float(np.mean(diag["A"])),
        "mean_safe_delta": float(np.mean(diag["safe_delta"])),
        "mean_sigma_i": float(np.mean(diag["sigma_i"])),
        "mean_alpha_i": float(np.mean(diag["alpha_i"])),
        "v_unbiased_norm": v_unbiased.norm().item(),
        "v_intentional_norm": v_intentional.norm().item(),
    }
    return cos_sim, out_diag


def run(checkpoint_path, out_raw_csv, n_states=1000, n_actions=1000, state_seed=None, debug=False):
    ckpt, actor, critic, frozen = load_checkpoint_full(checkpoint_path)
    env_name = ckpt["env_name"]
    seed = ckpt["seed"]
    gamma = ckpt["gamma"]
    obs_mean, obs_var, obs_eps = ckpt["obs_mean"], ckpt["obs_var"], ckpt["obs_epsilon"]
    reward_var, reward_eps = ckpt["reward_var"], ckpt["reward_epsilon"]

    if state_seed is None:
        state_seed = seed + 1_000_000

    torch.manual_seed(state_seed)
    np.random.seed(state_seed)

    states, env = collect_states(env_name, actor, obs_mean, obs_var, obs_eps, state_seed, n_states)
    raw_env = env.unwrapped
    act_low, act_high = env.action_space.low, env.action_space.high

    os.makedirs(os.path.dirname(out_raw_csv), exist_ok=True)
    t0 = time.time()
    with open(out_raw_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state_idx", "cos_sim", "mean_A", "mean_safe_delta", "mean_sigma_i",
                          "mean_alpha_i", "v_unbiased_norm", "v_intentional_norm"])
        for idx, (qpos, qvel) in enumerate(states):
            cos_sim, diag = measure_state_bias_full(
                raw_env, actor, critic, obs_mean, obs_var, obs_eps,
                reward_var, reward_eps, gamma, qpos, qvel, n_actions,
                act_low, act_high, frozen,
            )
            writer.writerow([idx, cos_sim, diag["mean_A"], diag["mean_safe_delta"], diag["mean_sigma_i"],
                              diag["mean_alpha_i"], diag["v_unbiased_norm"], diag["v_intentional_norm"]])
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
