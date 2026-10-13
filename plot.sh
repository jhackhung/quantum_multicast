#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# 一次畫出所有圖（dests / alpha / qc），y 軸範圍直接在下方表格調整
#
# 用法:
#   ./plot.sh                 # 畫全部
#   ./plot.sh dests           # 只畫某一類：dests | alpha | qc（可多個）
#   ./plot.sh alpha qc
#
# 每一列格式（空白分隔）:  <topology>  <metric>  <y_min>  <y_max>
#   topology : TATA | ITCD | synthetic_150（dests/alpha）；synthetic（qc）
#   metric   : total_cost | transmission_cost | computation_cost | computation_cost_ratio
#   y_min / y_max : 數字；填 - 表示不設定（自動）
# 沒列在表格裡的 (topology, metric) 不會被畫出來。
# =========================================================

EXP_DIR="experiment"
OUT_DIR="img"
STEP_DESTS=5            # dests 圖 x 軸刻度間距
DESTS_START="40"          # dests 圖 x 軸範圍（num_dests），留空 = 不限
DESTS_END="60"
ALPHA_FIXED_DESTS=50    # alpha 圖固定的 num_dests
ALPHA_START="7"          # alpha 圖 x 軸範圍（alpha 值），留空 = 不限
ALPHA_END="15"
FIXED_ALPHA=4           # dests / qc 圖使用的 alpha（對應 csv 檔名）
QC_START_NODE="150"        # qc 圖 |B| 範圍，留空 = 不限
QC_END_NODE="390"
QC_FIXED_DESTS=50
AUTO_BREAK=${AUTO_BREAK:-1}   # 1 = 啟用 --auto-break（自動拿掉沒有線經過的 y 區間）；也可 AUTO_BREAK=1 ./plot.sh 覆寫
AUTO_BREAK_METRICS="total_cost"   # 只對這些 metric 啟用 auto-break（空白分隔）；其餘走一般圖
MAX_SEGMENTS=${MAX_SEGMENTS:-2}   # auto-break 最多切成幾段（2 = 只切一刀）
LOWER_FRAC=${LOWER_FRAC:-0.65}    # 只切一刀時 QSTA 附近（下方）子圖佔的高度比例；設成空字串 = 依資料範圍自動分配

# ---------------------------------------------------------
# dests sweep：x = num_dests
# ---------------------------------------------------------
DESTS_LIMITS="
TATA            total_cost              -  -
TATA            transmission_cost       -  -
TATA            computation_cost        -  -
TATA            computation_cost_ratio  -  -
ITCD            total_cost              -  -
ITCD            transmission_cost       -  -
ITCD            computation_cost        -  -
ITCD            computation_cost_ratio  -  -
synthetic_150   total_cost              -  -
synthetic_150   transmission_cost       -  -
synthetic_150   computation_cost        -  -
synthetic_150   computation_cost_ratio  -  -
"

# ---------------------------------------------------------
# alpha sweep：x = alpha（固定 num_dests = ALPHA_FIXED_DESTS）
# ---------------------------------------------------------
ALPHA_LIMITS="
TATA            total_cost              -  -
TATA            transmission_cost       40  -
TATA            computation_cost        -  -
TATA            computation_cost_ratio  -  -
ITCD            total_cost              -  -
ITCD            transmission_cost       40  -
ITCD            computation_cost        -  -
ITCD            computation_cost_ratio  -  -
synthetic_150   total_cost              -  -
synthetic_150   transmission_cost       20  -
synthetic_150   computation_cost        -  -
synthetic_150   computation_cost_ratio  -  -
"

# ---------------------------------------------------------
# qc sweep：x = |B|（只有 synthetic）
# ---------------------------------------------------------
QC_LIMITS="
synthetic       total_cost              -  -
synthetic       transmission_cost       -  -
synthetic       computation_cost        -  -
synthetic       computation_cost_ratio  -  -
"

# ---------------------------------------------------------
# 以下為執行邏輯，一般不需修改
# ---------------------------------------------------------
PYTHON=$(command -v python3 || command -v python) || { echo "錯誤: 未找到 python"; exit 1; }

WHICH=("$@")
[ ${#WHICH[@]} -eq 0 ] && WHICH=(dests alpha qc)

limit_args() {  # $1=y_min $2=y_max $3=metric -> 設在全域陣列 LIM；並依 AUTO_BREAK 與 AUTO_BREAK_METRICS 設定全域陣列 AB
    LIM=()
    [ "$1" != "-" ] && LIM+=(--y-min "$1")
    [ "$2" != "-" ] && LIM+=(--y-max "$2")
    AB=()
    if [ "$AUTO_BREAK" = "1" ] && [[ " $AUTO_BREAK_METRICS " == *" $3 "* ]]; then
        AB+=(--auto-break --max-segments "$MAX_SEGMENTS")
        [ -n "$LOWER_FRAC" ] && AB+=(--lower-frac "$LOWER_FRAC")
    fi
    return 0
}

run_dests() {
    local range=()
    [ -n "$DESTS_START" ] && range+=(--x-min "$DESTS_START")
    [ -n "$DESTS_END" ] && range+=(--x-max "$DESTS_END")
    while read -r topo metric ymin ymax; do
        [ -z "${topo:-}" ] && continue
        limit_args "$ymin" "$ymax" "$metric"
        echo ">>> dests / $topo / $metric"
        "$PYTHON" plot_result.py "$EXP_DIR/dests_${topo}_alpha_${FIXED_ALPHA}_k_2.csv" \
            --x dests --step "$STEP_DESTS" --metrics "$metric" --out-dir "$OUT_DIR" "${range[@]}" "${LIM[@]}" "${AB[@]}"
    done <<< "$DESTS_LIMITS"
}

run_alpha() {
    local range=()
    [ -n "$ALPHA_START" ] && range+=(--x-min "$ALPHA_START")
    [ -n "$ALPHA_END" ] && range+=(--x-max "$ALPHA_END")
    while read -r topo metric ymin ymax; do
        [ -z "${topo:-}" ] && continue
        limit_args "$ymin" "$ymax" "$metric"
        # 以數字排序，確保 alpha 由小到大
        mapfile -t files < <(ls "$EXP_DIR"/dests_${topo}_alpha_*_k_2.csv | sort -V)
        echo ">>> alpha / $topo / $metric (${#files[@]} files)"
        "$PYTHON" plot_result.py "${files[@]}" \
            --x alpha --fixed-dests "$ALPHA_FIXED_DESTS" --metrics "$metric" --out-dir "$OUT_DIR" "${range[@]}" "${LIM[@]}" "${AB[@]}"
    done <<< "$ALPHA_LIMITS"
}

run_qc() {
    local range=()
    [ -n "$QC_START_NODE" ] && range+=(--x-min "$QC_START_NODE")
    [ -n "$QC_END_NODE" ] && range+=(--x-max "$QC_END_NODE")
    while read -r topo metric ymin ymax; do
        [ -z "${topo:-}" ] && continue
        limit_args "$ymin" "$ymax" "$metric"
        echo ">>> qc / $topo / $metric"
        "$PYTHON" plot_result.py "$EXP_DIR/qc_${topo}_alpha_${FIXED_ALPHA}_k_2_d${QC_FIXED_DESTS}.csv" \
            --x qc --metrics "$metric" --out-dir "$OUT_DIR" "${range[@]}" "${LIM[@]}" "${AB[@]}"
    done <<< "$QC_LIMITS"
}

for w in "${WHICH[@]}"; do
    case "$w" in
        dests) run_dests ;;
        alpha) run_alpha ;;
        qc)    run_qc ;;
        *) echo "錯誤: 不支援 '$w'（可用: dests alpha qc）"; exit 1 ;;
    esac
done
echo "Done."