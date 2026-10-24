#!/bin/bash
#SBATCH --job-name=qm_k
#SBATCH --account=acd109125
#SBATCH --partition=ct56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-79%25
#SBATCH --output=logs/k_%A_%a.out
#SBATCH --error=logs/k_%A_%a.err
#
# 100 seed x (nchc/sweep.json 的 sweep)，k 由環境變數 K 指定（預設 3）。
# 一個 array task = 一個 seed = 一個獨立 process/目錄 runs/k<K>_seed<SEED>/。
# （--mem 是估計值，請依 probe 的 peak RSS 調整，建議設 2 倍。）
#
# ct56 的 QOS：MaxSubmitPU=80、MaxJobsPU=25，所以 100 個 seed 分兩批送：
#   sbatch nchc/run_k.sh                      # task 0-79（seed = config.seed + 0~79）
#   sbatch --array=80-99%25 nchc/run_k.sh     # task 80-99（等第一批清空再送）
# 其他 k：K=1 sbatch nchc/run_k.sh
# 注意：logs/ 必須在 sbatch 之前就存在（mkdir -p logs），否則 Slurm 無法寫 log。

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source nchc/env.sh

K=${K:-3}
echo "=== start $(date -Is) | task=$SLURM_ARRAY_TASK_ID k=$K ==="
# 不用 srun：這是單一 task 的串列作業，srun 沒有好處，
# 反而會因為 SLURM_CPUS_PER_TASK / SLURM_TRES_PER_TASK 不一致而 fatal。
/usr/bin/time -v python nchc/run_seed.py --k "$K" --task-id "$SLURM_ARRAY_TASK_ID" --runs runs
echo "=== done  $(date -Is) | task=$SLURM_ARRAY_TASK_ID ==="
