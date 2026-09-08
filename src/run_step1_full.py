"""Step 1 pilot/full run for one (env, seed), using the RMSProp-inclusive
"full Algorithm 3" bias measurement (measure_bias_full.py) instead of the
Appendix-B-only version (measure_bias.py / run_step1.py).
"""
import argparse
import csv
import json
import os
import subprocess
import time

import numpy as np
import torch

from train_checkpoint import train_to_checkpoint
from measure_bias_full import run as measure_run_full


def git_hash(repo_dir):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def percentile(arr, p):
    return float(np.percentile(arr, p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Ant-v4")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--n_states", type=int, default=1000)
    parser.add_argument("--n_actions", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.8)
    parser.add_argument("--eta_policy", type=float, default=0.05)
    parser.add_argument("--eta_value", type=float, default=0.5)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--results_dir", type=str, default="../results")
    parser.add_argument("--tag", type=str, default="pilot_full")
    args = parser.parse_args()

    torch.set_num_threads(1)

    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    ghash = git_hash(project_dir)

    ckpt_dir = os.path.join(args.results_dir, "checkpoints_full")
    raw_dir = os.path.join(args.results_dir, f"step1_raw_{args.tag}")
    summary_dir = os.path.join(args.results_dir, f"step1_summary_{args.tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"{args.env_name}_seed{args.seed}_{args.total_steps}steps.pt")
    raw_csv = os.path.join(raw_dir, f"{args.env_name}_seed{args.seed}.csv")
    summary_json = os.path.join(summary_dir, f"{args.env_name}_seed{args.seed}.json")

    t0 = time.time()
    print(f"[run_step1_full] training {args.env_name} seed={args.seed} for {args.total_steps} steps...", flush=True)
    ckpt = train_to_checkpoint(
        env_name=args.env_name, seed=args.seed, gamma=args.gamma, lamda=args.lamda,
        total_steps=args.total_steps, entropy_coeff=args.entropy_coeff,
        eta_policy=args.eta_policy, eta_value=args.eta_value,
    )
    torch.save(ckpt, ckpt_path)
    train_time = time.time() - t0
    print(f"[run_step1_full] training done in {train_time:.1f}s, saved {ckpt_path}", flush=True)

    t1 = time.time()
    measure_run_full(ckpt_path, raw_csv, n_states=args.n_states, n_actions=args.n_actions, debug=True)
    measure_time = time.time() - t1
    print(f"[run_step1_full] measurement done in {measure_time:.1f}s, saved {raw_csv}", flush=True)

    cos_sims = []
    with open(raw_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = float(row["cos_sim"])
            if not np.isnan(v):
                cos_sims.append(v)
    cos_sims = np.array(cos_sims)

    summary = {
        "env_name": args.env_name,
        "seed": args.seed,
        "n_states": args.n_states,
        "n_actions": args.n_actions,
        "n_valid_states": int(len(cos_sims)),
        "mean": float(np.mean(cos_sims)),
        "std": float(np.std(cos_sims)),
        "median": float(np.median(cos_sims)),
        "pct20": percentile(cos_sims, 20),
        "pct5": percentile(cos_sims, 5),
        "pct1": percentile(cos_sims, 1),
        "gamma": args.gamma, "lamda": args.lamda,
        "eta_policy": args.eta_policy, "eta_value": args.eta_value,
        "entropy_coeff": args.entropy_coeff,
        "total_steps": args.total_steps,
        "git_hash": ghash,
        "method": "full_algorithm3_rmsprop",
        "train_wall_clock_sec": train_time,
        "measure_wall_clock_sec": measure_time,
        "checkpoint_path": ckpt_path,
        "raw_csv_path": raw_csv,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_step1_full] SUMMARY {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
