#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "用法: ./run.sh <次數> <config.json> [all|spt,clea,dmst,kmb,mfcs]"
    echo "範例:"
    echo "  ./run.sh 5 configs/tata.json"
    echo "  ./run.sh 5 configs/itcd.json kmb,mfcs"
    exit 1
fi

N=$1
CONFIG=$2
ALGO=${3:-all}

if ! command -v python3 &> /dev/null; then
    echo "錯誤: 未找到 python3"
    exit 1
fi

# =========================================================
# 從 config 讀實驗參數
#   base_dests: sweep 起點的 num_dests
#   step_dests: 每次遞增的 destination 數
# 皆已存在於本專案的 config schema 中 (見 configs/*.json)
# =========================================================
BASE_NDESTS=$(python3 -c "
import json
cfg = json.load(open('$CONFIG'))
print(cfg.get('num_dests', 30))
")
STEP_NDESTS=$(python3 -c "
import json
cfg = json.load(open('$CONFIG'))
print(cfg.get('step_dests', 5))
")

echo "N=$N"
echo "CONFIG=$CONFIG"
echo "ALGO=$ALGO"
echo "BASE_NDESTS=$BASE_NDESTS"
echo "STEP_NDESTS=$STEP_NDESTS"

# =========================================================
# 逐次遞增 num_dests，寫入暫存 config，呼叫 main.py
# main.py 會以 append 模式把每次的結果寫進同一份
# checkpoints/dests_<name>_alpha_<alpha>.csv，故不需清舊檔即可疊加畫圖
# =========================================================
run_dests() {
    echo "======================================"
    echo "開始跑 dests sweep | BASE_NDESTS=$BASE_NDESTS | STEP_NDESTS=$STEP_NDESTS"
    echo "======================================"

    for ((i = 1; i <= N; i++)); do
        NDESTS=$((BASE_NDESTS + STEP_NDESTS * (i - 1)))
        TMP_CONFIG=$(mktemp /tmp/config_dests.XXXXXX.json)

        python3 -c "
import json, sys
cfg = json.load(open('$CONFIG'))
cfg['num_dests'] = $NDESTS
cfg['algos'] = '$ALGO'
json.dump(cfg, open('$TMP_CONFIG', 'w'), indent=2, ensure_ascii=False)
"

        echo "=== dests | 第 $i 次 === num_dests=$NDESTS config=$TMP_CONFIG"

        python3 main.py "$TMP_CONFIG" dests "$ALGO"

        rm -f "$TMP_CONFIG"
    done
}

run_dests

echo "所有實驗執行完畢！"
