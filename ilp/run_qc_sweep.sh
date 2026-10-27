#!/usr/bin/env bash
# ILP 最佳解 qc sweep：|B|=10..100 step 10，alpha=4、k=2，|D| 由參數指定
# 用法: bash ilp/run_qc_sweep.sh [dests] [time_limit] [runs] [b_sizes]
# 輸出: ilp/results/qc_sweep_d<dests>.csv
# 只跑部分 |B|：BS="80 90 100" bash ilp/run_qc_sweep.sh 30 300 100
# 直譯器：環境變數 PYTHON > ilp/.venv（Windows 或 Linux 版）> PATH 上的 python
D=${1:-10}; TL=${2:-300}; RUNS=${3:-1}; BS=${4:-"10 20 30 40 50 60 70 80 90 100"}
cd "$(dirname "$0")/.."
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON
elif [ -x ilp/.venv/Scripts/python.exe ]; then PY=ilp/.venv/Scripts/python.exe
elif [ -x ilp/.venv/bin/python ]; then PY=ilp/.venv/bin/python
else PY=python; fi
echo "using: $PY"
for n in ${BS:-10 20 30 40 50 60 70 80 90 100}; do
  "$PY" -m ilp.run_experiments --config ilp/configs/qc_$n.json --dests $D --alphas 4 --k 2 \
    --runs $RUNS --time-limit $TL --hint --out ilp/results/qc_sweep_d$D.csv
done
