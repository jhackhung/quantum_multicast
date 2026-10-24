#!/bin/bash
#SBATCH --job-name=qm_smoke
#SBATCH --account=acd109125
#SBATCH --partition=ctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#
# 冒煙測試（ctest，上限 2 小時）：小規模、k=1、2 個 seed，確認環境/路徑/輸出與 merge 都正常。
# 跑完檢查：
#   ls runs/smoke_seed*/raw/*.json
#   cat merged_smoke/all_aggregated.csv

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source nchc/env.sh

# 用 sweep.json 的 sweep 類型，但值縮到最小的一個，只跑便宜的 algo
SMALL=$(python -c "import json; print(min(json.load(open('nchc/sweep.json'))['values']))")
for T in 0 1; do
    python nchc/run_seed.py --k 1 --task-id "$T" --runs runs --prefix smoke \
        --values "$SMALL" --algos spt,clea,dmst,kmb,mfcs
done
python nchc/merge_results.py --runs runs --k 1 --prefix smoke --out merged_smoke
echo "=== smoke ok $(date -Is) ==="
