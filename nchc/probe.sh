#!/bin/bash
#SBATCH --job-name=qm_probe
#SBATCH --account=acd109125
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err
#
# 探測：1 seed、k 由環境變數 K 指定（預設 2）、sweep 裡「最大」的那個值（最壞情況），量 wall time 與 peak RSS，
# 用來估 run_k.sh 的 --time / --mem。k 越大越慢（k=3 遠比 k=2 慢），要估 k=3 就用 K=3 跑；
# 時間不夠時在命令列覆寫，例如: K=3 sbatch --time=4-00:00:00 nchc/probe.sh
# 讀法: grep -E "Elapsed \(wall clock\)|Maximum resident" logs/probe_*.err
# 注意：probe 也占 ct56 的 MaxSubmitPU 名額。

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source nchc/env.sh

# 最大的 sweep 值
K=${K:-2}
MAXV=$(python -c "import json; print(max(json.load(open('nchc/sweep.json'))['values']))")
echo "=== probe start $(date -Is) | value=$MAXV k=$K ==="
/usr/bin/time -v python nchc/run_seed.py --k "$K" --task-id 0 --runs runs --prefix "probe_k$K" --values "$MAXV"
echo "=== probe done  $(date -Is) ==="
