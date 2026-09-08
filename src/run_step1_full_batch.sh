#!/bin/bash
set -e
cd "$(dirname "$0")/src"
seq 3 29 | xargs -P 8 -I{} bash -c '
  seed="$1"
  PYTHONPATH=../baseline ../.venv/bin/python run_step1_full.py --env_name Ant-v4 --seed "$seed" --results_dir ../results --tag full > "../results/_logs/seed${seed}.log" 2>&1
  echo "seed $seed done"
' _ {}
echo "ALL_30_SEEDS_DONE"
