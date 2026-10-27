# ilp/ — WQMN 的 CP-SAT 最佳解與方法比較

論文：*Cost-Aware Multi-Copy Quantum Multicast with Lossless Quantum Data Compression*（WQMN / QSTA）。
本目錄用 OR-Tools CP-SAT 對同一個問題求最佳解（或時限內的 incumbent + best bound），
並與 SPT / CLEA / DMST / KMB / MFCS / RAW / TBQCM 比較。**不修改 repo 既有檔案**，只 import 呼叫。

## 環境（venv 在 ilp/.venv）

```
python -m venv ilp/.venv
ilp\.venv\Scripts\activate
pip install -r ilp/requirements.txt
```

以下指令都在 repo 根目錄執行。

## 方法

| 名稱 | 內容 |
|---|---|
| SPT/CLEA/DMST/KMB/MFCS | main.py 的 baseline，成本由 evaluate.py 計 |
| RAW | PDQTA 的輸出直接評分（不做 TBQCM） |
| TBQCM | PDQTA + TBQCM = 論文完整 QSTA（main.py 裡的 "QSTA"） |
| ILP | 本目錄的 CP-SAT 模型（model.py） |

## 模型（model.py）

變數 Q_e、P_e、b_u、y_{e,j}；限制式為論文 (6)–(15)：Q 守恆、P ≤ Q、P ≥ ⌈log₂(Q+1)⌉（以 y 精確表達）、
b_u = 0 ⇒ 進出 P 守恆。s 沒有入邊，但 b_s = 0 ⇒ Σ_out P = |D|（s 一次注入 |D| 個原始 qubit）。
D 只收不送；不要求 in-degree ≤ 1（論文無此限制）。目標 `Σ W·P + α·Σ b`，W、α 以 1e6 整數縮放，
**回報的成本一律用浮點公式從解重算**；縮放誤差上界記在 `round_err_bound`（約 1e-3 級）。

## loose vs strict（兩種計價，可比較）

- **loose**：各方法原本的評分（evaluate.py / QSTA.evaluate_qsta_tree）。
- **strict**：保留方法自己的 Q/P/b，只替「依 (12)–(15) 必須啟用卻沒啟用」的節點補 b=1 並加 α。
  CP-SAT 的解本身就是 strict。

evaluate.apply_lqdc_placement_baseline 只看「候選節點」就把進入該節點的邊算成壓縮量，不要求送出端也啟用，
因此 baseline 的 loose 成本可能低於論文限制下的真實成本（甚至低於最佳解）。
`compare.py` / CSV 同時輸出兩種：`cost_loose`、`cost_strict`、`gap_loose`、`gap_strict`、
`extra_activations`（strict 多補的啟用數）。QSTA 系列（RAW/TBQCM）的 loose = strict。

另外 CLEA 的樹含 Q=0 的死分支但 evaluate.py 仍計費，欄位 `dead_edges` 記錄其數量，
`feasible_strict=False` 即來自此。

## 使用

```
# 單一實例比較（Python）
from ilp.instance import build_instance, load_config
from ilp.compare import compare_instance
qn = build_instance(load_config("configs/itcd.json"), run_idx=0, num_dests=10)
rows = compare_instance(qn, alpha=4, time_limit=60, use_hint=True)

# 批次（可斷點續跑）
python -m ilp.run_experiments --config configs/itcd.json --dests 5,10,15,20 \
    --alphas 4 --runs 5 --time-limit 600 --hint --out ilp/results/itcd.csv

# 圖：method/OPT，loose 與 strict 並排；空心點 = 該點 CP-SAT 未證明最佳
python -m ilp.plot_gap ilp/results/itcd.csv --mode both --ymax 2.5 --out ilp/results/itcd.png

# 測試
python -m pytest ilp/tests -q
```

seed 規則同 main.py：`seed = cfg.seed + run`，`role_seed = cfg.role_seed + run`；CP-SAT `random_seed = run`。

## CSV 欄位

實例：`name num_nodes num_B num_edges num_dests alpha k run seed time_limit hint`。
每列一個方法：`method status runtime cost_loose cost_strict trans_strict comp_strict extra_activations
dead_edges feasible_loose feasible_strict gap_loose gap_strict gap_vs_bound_* opt opt_bound opt_status
best_bound solver_gap round_err_bound note`。

- `gap_* = (method − OPT)/OPT`，OPT 取 CP-SAT incumbent（最小化問題，gap ≥ 0 才合理）。
- `opt_status != OPTIMAL` 時 OPT 只是上界，真實 gap 的上界看 `gap_vs_bound_*`（以 best bound 為分母）。

## 驗證（ilp/tests）

- 論文 Fig.1：不壓縮成本 40；Fig.2（α=0.5）最佳值 60.5，與論文 Example 1 一致。
- 獨立的 scipy/HiGHS MILP（big-M 編碼）與 CP-SAT 最佳值逐例比對（sample 與隨機小實例）。
- strict 計價下所有啟發式 ≥ OPT；驗證器能抓出守恆、P<⌈log₂(Q+1)⌉、缺 b 的違規。
- TBQCM 轉換後的成本與 `QSTA.build_and_evaluate_qsta` 完全一致。

## 規模（實測，ITCD |V|≈118–133、|E|≈330、60 秒、8 workers）

|D| = 5：秒級 OPTIMAL；|D| ≥ 10：60 秒內只有 incumbent，bound gap 約 15–30%。
需要證明最佳性時請提高 time limit 或縮小實例；`--hint`（用 TBQCM 解當 hint）能改善 incumbent。

## qc sweep 操作流程（|B| = 10..100）

以下指令都在 repo 根目錄、Git Bash 執行。

```
# 跑實驗：bash ilp/run_qc_sweep.sh [dests] [time_limit 秒] [runs]
bash ilp/run_qc_sweep.sh 30 300 1      # 輸出 ilp/results/qc_sweep_d30.csv
bash ilp/run_qc_sweep.sh 50 300 5      # 每點 5 個 run，可算平均與誤差棒

# 繪圖
ilp/.venv/Scripts/python -m ilp.plot_gap ilp/results/qc_sweep_d30.csv \
    --x num_B --mode strict --out ilp/results/qc_sweep_d30.png
```

- 腳本依序跑 |B|=10..100，alpha=4、k=2 寫死在 `ilp/run_qc_sweep.sh`；每個 |B| 的拓撲在 `ilp/configs/qc_<n>.json`
  （`area_size = 20·√(|B|/150)`，`allow_repeat_hosts=true`）。
- 斷點續跑 key 是 (name, dests, alpha, run)，每個 |B| 的 config name 不同（`ilp_qc_<n>`）才不會互相略過；
  中斷後重跑同一指令即可，要重跑某設定先刪對應 CSV。
- **不同設定請寫不同 CSV**（檔名帶 `_d<dests>`），不要共用同一檔。
- 中途要停請直接結束 bash 與 python 程序，只停外層 shell 會讓迴圈繼續跑並覆寫 CSV。
- `--x` 可選 `num_dests / alpha / num_nodes / num_B`；`--mode` 可選 `strict / loose / both`。

### 補跑：只重跑沒證明最佳的 run（不動原 CSV）

```
# 1. 補完被中斷的 |B|（續跑，已完成的 run 會 skip）
BS="80" bash ilp/run_qc_sweep.sh 30 300 100

# 2. 只重跑 status != OPTIMAL 的 run，時限拉高；結果寫 *_retry.csv，並產生 *_merged.csv
ilp/.venv/Scripts/python -m ilp.retry_nonopt ilp/results/qc_sweep_d30.csv --time-limit 900
#    試跑：--B 90,100 --max 3；中斷後只合併：--merge-only

# 3. 畫圖讀合併後的檔案
ilp/.venv/Scripts/python -m ilp.plot_gap ilp/results/qc_sweep_d30_merged.csv --x num_B --mode strict --out ilp/results/qc_sweep_d30_merged.png
```

合併時，retry 的 ILP 成本不比原本差才換掉整組列，否則保留原本。步驟 1 要先做，否則 |B|=80 新跑出的非 OPTIMAL run 不會被步驟 2 抓到。
