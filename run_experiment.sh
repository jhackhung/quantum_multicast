#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# 一次跑完某個 topology 的所有 sweep（dests / qc / alpha）
#
# 用法:
#   ./run_experiment.sh <target> [sweeps] [algos] [次數]
#     target : all | tata | itcd | synthetic
#     sweeps : all | dests,qc,alpha 的任意組合（預設 all）
#     algos  : all | spt,clea,dmst,kmb,mfcs,qsta（預設 all）
#     次數   : sweep 的點數，即 num_dests / num_nodes 遞增幾次（預設 5）
#
# 範例:
#   ./run_experiment.sh all
#   ./run_experiment.sh tata dests
#   ./run_experiment.sh synthetic qc,alpha qsta,mfcs 6
#
# alpha 不改 configs/*.json，直接用下面 ALPHA_* 變數覆寫進暫存 config。
# =========================================================

# ---------------------------------------------------------
# alpha 設定（在此調整，不動 configs/*.json）
#   SWEEP_ALPHAS : alpha sweep 要掃的 alpha 清單
#   FIXED_ALPHA  : 跑 dests / qc sweep 時固定使用的單一 alpha
# ---------------------------------------------------------
SWEEP_ALPHAS_TATA="1, 2, 4, 5, 6, 7, 8, 9, 10"
SWEEP_ALPHAS_ITCD="1, 2, 4, 5, 6, 7, 8, 9, 10"
SWEEP_ALPHAS_SYNTHETIC="1, 2, 4, 5, 6, 7, 8, 9, 10"
# SWEEP_ALPHAS_TATA="11, 12, 13, 14, 15, 16, 17, 18, 19, 20"
# SWEEP_ALPHAS_ITCD="11, 12, 13, 14, 15, 16, 17, 18, 19, 20"
# SWEEP_ALPHAS_SYNTHETIC="11, 12, 13, 14, 15, 16, 17, 18, 19, 20"

FIXED_ALPHA_TATA="3"
FIXED_ALPHA_ITCD="3"
FIXED_ALPHA_SYNTHETIC="3"

DEST_SWEEP_START=20
DEST_SWEEP_STEP=5

# ---------------------------------------------------------
# 每個 target 對應的 config
#   *_DESTS_CONFIG : dests sweep 與 alpha sweep 用
#   *_QC_CONFIG    : qc (num_nodes / |B|) sweep 用，僅 synthetic 有
# ---------------------------------------------------------
TATA_DESTS_CONFIG="configs/tata.json"
ITCD_DESTS_CONFIG="configs/itcd.json"
SYNTHETIC_DESTS_CONFIG="configs/synthetic_dests.json"
SYNTHETIC_QC_CONFIG="configs/synthetic_qc.json"

TARGET=${1:-all}
SWEEPS=${2:-all}
ALGO=${3:-all}
N=${4:-5}

usage() {
    echo "用法: ./run_experiment.sh <all|tata|itcd|synthetic> [all|dests,qc,alpha] [all|spt,clea,...] [次數]"
    exit 1
}

case "$TARGET" in
    all|tata|itcd|synthetic) ;;
    *) echo "錯誤: 不支援的 target '$TARGET'"; usage ;;
esac

if ! command -v python3 &> /dev/null; then
    echo "錯誤: 未找到 python3"
    exit 1
fi

if [ "$SWEEPS" = "all" ]; then
    SWEEPS="dests,qc,alpha"
fi

has_sweep() {
    # 判斷 $1 是否在逗號分隔的 SWEEPS 清單中
    [[ ",$SWEEPS," == *",$1,"* ]]
}

for s in ${SWEEPS//,/ }; do
    case "$s" in
        dests|qc|alpha) ;;
        *) echo "錯誤: 不支援的 sweep '$s'（僅支援 dests / qc / alpha）"; usage ;;
    esac
done

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# write_config <來源 config> <輸出路徑> <alpha 字面值> [key=value ...]
# alpha 以 python 字面值傳入（"2" 或 "[2, 2.5, 3]"），故可同時支援單值與清單。
write_config() {
    local SRC=$1 DST=$2 ALPHA_LITERAL=$3
    shift 3
    local OVERRIDES=("$@")

    python3 - "$SRC" "$DST" "$ALPHA_LITERAL" "${OVERRIDES[@]}" << 'PY'
import ast, json, sys

src, dst, alpha_literal = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(src, encoding="utf-8"))
cfg["alpha"] = ast.literal_eval(alpha_literal)

for item in sys.argv[4:]:
    key, _, value = item.partition("=")
    cfg[key] = ast.literal_eval(value)

json.dump(cfg, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
PY
}

cfg_get() {
    # cfg_get <config> <key> <fallback-expr>
    python3 -c "
import json, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8'))
print(cfg.get(sys.argv[2], $3))
" "$1" "$2"
}

# =========================================================
# dests sweep: 逐次遞增 num_dests，alpha 固定為單一值
# 結果疊加至 checkpoints/dests_<name>_alpha_<alpha>_n<n>_d<d>_k_<k>.json
# =========================================================
run_dests() {
    local LABEL=$1 CONFIG=$2 ALPHA=$3
    local BASE STEP NDESTS TMP_CONFIG

    BASE=$DEST_SWEEP_START
    STEP=$DEST_SWEEP_STEP

    echo ""
    echo "######################################################################"
    echo "# [$LABEL] dests sweep | config=$CONFIG | alpha=$ALPHA"
    echo "# base=$BASE step=$STEP points=$N algos=$ALGO"
    echo "######################################################################"

    for ((i = 1; i <= N; i++)); do
        NDESTS=$((BASE + STEP * (i - 1)))
        TMP_CONFIG="$TMP_DIR/${LABEL}_dests_${NDESTS}.json"
        write_config "$CONFIG" "$TMP_CONFIG" "$ALPHA" "num_dests=$NDESTS"

        echo ""
        echo "=== [$LABEL] dests | 第 $i/$N 次 | num_dests=$NDESTS alpha=$ALPHA ==="
        python3 main.py "$TMP_CONFIG" dests "$ALGO"
    done
}

# =========================================================
# qc sweep: 逐次遞增 num_nodes（等於改變 |B| = num_nodes - num_dests），
# num_dests 維持 config 原值，alpha 固定為單一值。僅 synthetic 支援。
# =========================================================
run_qc() {
    local LABEL=$1 CONFIG=$2 ALPHA=$3
    local BASE STEP NNODES TMP_CONFIG BASE_AREA AREA_SIZE

    BASE=$(cfg_get "$CONFIG" base_qc "cfg.get('num_nodes', 150)")
    STEP=$(cfg_get "$CONFIG" step_qc 50)
    BASE_AREA=$(cfg_get "$CONFIG" area_size "cfg.get('area_size', 20.0)")

    echo ""
    echo "######################################################################"
    echo "# [$LABEL] qc (num_nodes) sweep | config=$CONFIG | alpha=$ALPHA"
    echo "# base=$BASE step=$STEP points=$N algos=$ALGO base_area=$BASE_AREA"
    echo "######################################################################"

    for ((i = 1; i <= N; i++)); do
        NNODES=$((BASE + STEP * (i - 1)))
        AREA_SIZE=$(python3 -c "import math; print($BASE_AREA * math.sqrt($NNODES / $BASE))")
        # AREA_SIZE=$(python3 -c "print($BASE_AREA)")

        TMP_CONFIG="$TMP_DIR/${LABEL}_qc_${NNODES}.json"
        write_config "$CONFIG" "$TMP_CONFIG" "$ALPHA" "num_nodes=$NNODES" "area_size=$AREA_SIZE"

        echo ""
        echo "=== [$LABEL] qc | 第 $i/$N 次 | num_nodes=$NNODES area_size=$AREA_SIZE alpha=$ALPHA ==="
        python3 main.py "$TMP_CONFIG" qc "$ALGO"
    done
}

# =========================================================
# alpha sweep: num_dests / num_nodes 維持 config 原值，只掃 alpha。
# main.py 本身會 for alpha in alpha_list，故一次呼叫即可跑完整組 alpha，
# 每個 alpha 各自寫到自己的 checkpoint / experiment csv。
# =========================================================
run_alpha() {
    local LABEL=$1 CONFIG=$2 ALPHAS=$3
    local TMP_CONFIG="$TMP_DIR/${LABEL}_alpha.json"

    write_config "$CONFIG" "$TMP_CONFIG" "[$ALPHAS]"

    echo ""
    echo "######################################################################"
    echo "# [$LABEL] alpha sweep | config=$CONFIG | alphas=[$ALPHAS]"
    echo "# algos=$ALGO"
    echo "######################################################################"

    python3 main.py "$TMP_CONFIG" dests "$ALGO"
}

run_target() {
    local LABEL=$1 DESTS_CONFIG=$2 QC_CONFIG=$3 FIXED_ALPHA=$4 SWEEP_ALPHAS=$5

    if has_sweep dests; then
        run_dests "$LABEL" "$DESTS_CONFIG" "$FIXED_ALPHA"
    fi

    if has_sweep qc; then
        if [ -n "$QC_CONFIG" ]; then
            run_qc "$LABEL" "$QC_CONFIG" "$FIXED_ALPHA"
        else
            echo ""
            echo "[$LABEL] 略過 qc sweep：qc sweep 只支援 synthetic（real topology 的 num_nodes 由 gml 決定）"
        fi
    fi

    if has_sweep alpha; then
        run_alpha "$LABEL" "$DESTS_CONFIG" "$SWEEP_ALPHAS"
    fi
}

# =========================================================
# 實驗計畫預覽：在真正執行前列出所有要跑的實驗與參數
# =========================================================
plan_target() {
    local LABEL=$1 DESTS_CONFIG=$2 QC_CONFIG=$3 FIXED_ALPHA=$4 SWEEP_ALPHAS=$5
    local MODE TOPO K RUNS SEED NDESTS_CFG NNODES_CFG
    local i v LIST

    MODE=$(cfg_get "$DESTS_CONFIG" mode "'?'")
    K=$(cfg_get "$DESTS_CONFIG" pdqta_level 2)
    RUNS=$(cfg_get "$DESTS_CONFIG" num_runs 1)
    SEED=$(cfg_get "$DESTS_CONFIG" seed 0)
    NDESTS_CFG=$(cfg_get "$DESTS_CONFIG" num_dests 0)
    if [ "$MODE" = "synthetic" ]; then
        TOPO="synthetic (num_nodes=$(cfg_get "$DESTS_CONFIG" num_nodes 100), area_size=$(cfg_get "$DESTS_CONFIG" area_size 20.0))"
    else
        TOPO="real ($(cfg_get "$DESTS_CONFIG" gml_path "'?'"))"
    fi

    echo "[$LABEL] topology = $TOPO"
    echo "        k(pdqta_level)=$K  num_runs=$RUNS  base_seed=$SEED  algos=$ALGO"

    if has_sweep dests; then
        LIST=""
        for ((i = 0; i < N; i++)); do
            LIST+="$((DEST_SWEEP_START + DEST_SWEEP_STEP * i)) "
        done
        echo "  - dests sweep : config=$DESTS_CONFIG"
        echo "      num_dests  = ${LIST% } (start=$DEST_SWEEP_START, step=$DEST_SWEEP_STEP, $N 點)"
        echo "      alpha(固定) = $FIXED_ALPHA"
    fi

    if has_sweep qc; then
        if [ -n "$QC_CONFIG" ]; then
            local BASE STEP AREA
            BASE=$(cfg_get "$QC_CONFIG" base_qc "cfg.get('num_nodes', 150)")
            STEP=$(cfg_get "$QC_CONFIG" step_qc 50)
            AREA=$(cfg_get "$QC_CONFIG" area_size 20.0)
            LIST=""
            for ((i = 0; i < N; i++)); do
                LIST+="$((BASE + STEP * i)) "
            done
            echo "  - qc sweep    : config=$QC_CONFIG"
            echo "      num_nodes  = ${LIST% } (base=$BASE, step=$STEP, $N 點)"
            echo "      num_dests  = $(cfg_get "$QC_CONFIG" num_dests 0) (固定), area_size=$AREA (固定)"
            echo "      alpha(固定) = $FIXED_ALPHA"
        else
            echo "  - qc sweep    : 略過（僅 synthetic 支援）"
        fi
    fi

    if has_sweep alpha; then
        echo "  - alpha sweep : config=$DESTS_CONFIG"
        echo "      alpha      = [$SWEEP_ALPHAS]"
        echo "      num_dests  = $NDESTS_CFG (固定，取自 config)"
    fi
    echo ""
}

print_plan() {
    echo "======================================================================"
    echo "實驗計畫（TARGET=$TARGET  SWEEPS=$SWEEPS  ALGO=$ALGO  N=$N）"
    echo "======================================================================"
    if [ "$TARGET" = "tata" ] || [ "$TARGET" = "all" ]; then
        plan_target TATA "$TATA_DESTS_CONFIG" "" "$FIXED_ALPHA_TATA" "$SWEEP_ALPHAS_TATA"
    fi
    if [ "$TARGET" = "itcd" ] || [ "$TARGET" = "all" ]; then
        plan_target ITCD "$ITCD_DESTS_CONFIG" "" "$FIXED_ALPHA_ITCD" "$SWEEP_ALPHAS_ITCD"
    fi
    if [ "$TARGET" = "synthetic" ] || [ "$TARGET" = "all" ]; then
        plan_target SYNTHETIC "$SYNTHETIC_DESTS_CONFIG" "$SYNTHETIC_QC_CONFIG"             "$FIXED_ALPHA_SYNTHETIC" "$SWEEP_ALPHAS_SYNTHETIC"
    fi
    echo "(已有 checkpoint 的 run 會自動 skip；設 YES=1 可略過確認)"
    echo "======================================================================"
}

print_plan
if [ -t 0 ] && [ "${YES:-0}" != "1" ]; then
    read -r -p "確認開始執行？[Y/n] " REPLY
    case "${REPLY:-Y}" in
        [Yy]*) ;;
        *) echo "已取消"; exit 0 ;;
    esac
fi

echo "TARGET=$TARGET"
echo "SWEEPS=$SWEEPS"
echo "ALGO=$ALGO"
echo "N=$N"

if [ "$TARGET" = "tata" ] || [ "$TARGET" = "all" ]; then
    run_target TATA "$TATA_DESTS_CONFIG" "" "$FIXED_ALPHA_TATA" "$SWEEP_ALPHAS_TATA"
fi

if [ "$TARGET" = "itcd" ] || [ "$TARGET" = "all" ]; then
    run_target ITCD "$ITCD_DESTS_CONFIG" "" "$FIXED_ALPHA_ITCD" "$SWEEP_ALPHAS_ITCD"
fi

if [ "$TARGET" = "synthetic" ] || [ "$TARGET" = "all" ]; then
    run_target SYNTHETIC "$SYNTHETIC_DESTS_CONFIG" "$SYNTHETIC_QC_CONFIG" \
        "$FIXED_ALPHA_SYNTHETIC" "$SWEEP_ALPHAS_SYNTHETIC"
fi

echo ""
echo "所有實驗執行完畢！"
