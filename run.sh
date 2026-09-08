#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "用法: ./run.sh <次數> <dests|qc> <config.json> [all|spt,clea,dmst,kmb,mfcs]"
    echo "範例:"
    echo "  ./run.sh 5 dests configs/tata.json"
    echo "  ./run.sh 5 dests configs/itcd.json kmb,mfcs"
    echo "  ./run.sh 5 qc configs/synthetic_qc.json"
    exit 1
fi

N=$1
SWEEP=$2
CONFIG=$3
ALGO=${4:-all}

if ! command -v python3 &> /dev/null; then
    echo "錯誤: 未找到 python3"
    exit 1
fi

case "$SWEEP" in
    dests|qc) ;;
    *)
        echo "錯誤: 不支援的 sweep 類型 '$SWEEP'（僅支援 dests 或 qc）"
        exit 1
        ;;
esac

echo "N=$N"
echo "SWEEP=$SWEEP"
echo "CONFIG=$CONFIG"
echo "ALGO=$ALGO"

# =========================================================
# 逐次遞增 num_dests，寫入暫存 config，呼叫 main.py
# main.py 會以 append 模式把每次的結果寫進同一份
# checkpoints/dests_<name>_alpha_<alpha>_k_<k>.json，故不需清舊檔即可疊加畫圖
# =========================================================
run_dests() {
    # base_dests: sweep 起點的 num_dests
    # step_dests: 每次遞增的 destination 數
    local BASE_NDESTS STEP_NDESTS
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

# =========================================================
# 逐次遞增 num_nodes（進而改變 |B| = num_nodes - num_dests），
# 寫入暫存 config，呼叫 main.py。num_dests 維持 config 原值不變。
# 結果疊加寫入 checkpoints/qc_<name>_alpha_<alpha>_k_<k>.json
# =========================================================
run_qc() {
    # base_qc: sweep 起點的 num_nodes
    # step_qc: 每次遞增的 node 數
    local BASE_NNODES STEP_NNODES
    BASE_NNODES=$(python3 -c "
import json
cfg = json.load(open('$CONFIG'))
print(cfg.get('base_qc', cfg.get('num_nodes', 150)))
")
    STEP_NNODES=$(python3 -c "
import json
cfg = json.load(open('$CONFIG'))
print(cfg.get('step_qc', 50))
")

    echo "======================================"
    echo "開始跑 qc (num_nodes) sweep | BASE_NNODES=$BASE_NNODES | STEP_NNODES=$STEP_NNODES"
    echo "======================================"

    for ((i = 1; i <= N; i++)); do
        NNODES=$((BASE_NNODES + STEP_NNODES * (i - 1)))
        TMP_CONFIG=$(mktemp /tmp/config_qc.XXXXXX.json)

        python3 -c "
import json, sys
cfg = json.load(open('$CONFIG'))
cfg['num_nodes'] = $NNODES
cfg['algos'] = '$ALGO'
json.dump(cfg, open('$TMP_CONFIG', 'w'), indent=2, ensure_ascii=False)
"

        echo "=== qc | 第 $i 次 === num_nodes=$NNODES config=$TMP_CONFIG"

        python3 main.py "$TMP_CONFIG" qc "$ALGO"

        rm -f "$TMP_CONFIG"
    done
}

if [ "$SWEEP" = "dests" ]; then
    run_dests
else
    run_qc
fi

echo "所有實驗執行完畢！"
