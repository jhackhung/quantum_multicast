#!/usr/bin/env bash
# 載入 Python 環境（被 submit.sh / *.sbatch source）。
# 依 NCHC 非互動式 shell 的限制：要先 source conda.sh 才能 conda activate。
# 環境名稱用 CONDA_ENV 指定（預設 qmulticast，由 setup_env.sh 建立）。
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0   # set 迭代順序固定，同一組 seed 可重現
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # 每個 task 只用 1 core

module purge
module load miniconda3
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-qmulticast}"
python -c 'import networkx, matplotlib' \
    || { echo "ERROR: conda 環境 ${CONDA_ENV:-qmulticast} 未就緒或缺套件（先跑 nchc/setup_env.sh）" >&2; exit 1; }
