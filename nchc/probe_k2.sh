#!/bin/bash
#SBATCH --job-name=qm_probe
#SBATCH --account=acd109125
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/probe_k2_%j.out
#SBATCH --error=logs/probe_k2_%j.err
#
# 探測：1 seed、k=2、sweep 裡「最大」的那個值（最壞情況），量 wall time 與 peak RSS，
# 用來估 run_k.sh 的 --time / --mem。k=3 比 k=2 更慢，這只是下限。
# 讀法: grep -E "Elapsed \(wall clock\)|Maximum resident" logs/probe_k2_*.err
# 注意：probe 也占 ct56 的 MaxSubmitPU 名額。

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source nchc/env.sh

# 最大的 sweep 值
MAXV=$(python -c "import json; print(max(json.load(open('nchc/sweep.json'))['values']))")
echo "=== probe start $(date -Is) | value=$MAXV k=2 ==="
/usr/bin/time -v python nchc/run_seed.py --k 2 --task-id 0 --runs runs --prefix probe_k2 --values "$MAXV"
echo "=== probe done  $(date -Is) ==="
