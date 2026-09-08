"""Train Intentional AC (unmodified baseline algorithm) to an exact step count and
save a checkpoint (network weights + frozen normalization statistics) for later
bias measurement (Step 1) or return comparison (Step 3).

Reuses Actor/Critic/IntentionalAC/optimizers verbatim from baseline/ -- no
algorithmic modification happens here.
"""
import argparse
import os
import time

import numpy as np
import torch

from common import make_training_env, IntentionalAC


def train_to_checkpoint(env_name, seed, gamma=0.99, lamda=0.8, total_steps=1_000_000,
                         entropy_coeff=0.01, eta_policy=0.05, eta_value=0.5,
                         hidden_size=128, log_every=None, debug=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_training_env(env_name, gamma)
    agent = IntentionalAC(
        n_obs=env.observation_space.shape[0],
        n_actions=env.action_space.shape[0],
        hidden_size=hidden_size,
        gamma=gamma, lamda=lamda, eta_policy=eta_policy, eta_value=eta_value,
    )

    returns_log = []
    s, _ = env.reset(seed=seed)
    t0 = time.time()
    for t in range(1, total_steps + 1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        agent.update_params(s, a, r, s_prime, done, terminated, entropy_coeff)
        s = s_prime
        if done:
            ep_ret = info["episode"]["r"]
            returns_log.append((t, float(ep_ret)))
            s, _ = env.reset()
            if debug and (log_every is None or len(returns_log) % log_every == 0):
                elapsed = time.time() - t0
                print(f"[{env_name} seed={seed}] step={t} episodic_return={ep_ret} elapsed={elapsed:.1f}s")

    checkpoint = {
        "env_name": env_name,
        "seed": seed,
        "gamma": gamma,
        "lamda": lamda,
        "total_steps": total_steps,
        "entropy_coeff": entropy_coeff,
        "eta_policy": eta_policy,
        "eta_value": eta_value,
        "hidden_size": hidden_size,
        "n_obs": env.observation_space.shape[0],
        "n_actions": env.action_space.shape[0],
        "policy_state_dict": agent.policy_net.state_dict(),
        "value_state_dict": agent.value_net.state_dict(),
        "obs_mean": env.obs_stats.mean.copy(),
        "obs_var": env.obs_stats.var.copy(),
        "obs_epsilon": env.epsilon,
        "reward_var": env.env.reward_stats.var.copy() if np.ndim(env.env.reward_stats.var) else float(env.env.reward_stats.var),
        "reward_epsilon": env.env.epsilon,
        "returns_log": returns_log,
        "train_wall_clock_sec": time.time() - t0,
    }
    env.close()
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Ant-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.8)
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--eta_policy", type=float, default=0.05)
    parser.add_argument("--eta_value", type=float, default=0.5)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(1)

    ckpt = train_to_checkpoint(
        env_name=args.env_name, seed=args.seed, gamma=args.gamma, lamda=args.lamda,
        total_steps=args.total_steps, entropy_coeff=args.entropy_coeff,
        eta_policy=args.eta_policy, eta_value=args.eta_value,
        log_every=args.log_every, debug=args.debug,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"Saved checkpoint to {args.out} (train_wall_clock_sec={ckpt['train_wall_clock_sec']:.1f})")
