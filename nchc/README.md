# 在台灣杉三號跑 quantum multicast 的 k=3（及 k=1）sweep 實驗

目標：依 [sweep.json](sweep.json) 的 sweep（預設 qc：num_nodes = 150/210/270/330/390，alpha=4），
100 個 seed，PDQTA k=3。

一個 Slurm array task = 一個 seed = 一個獨立 process 與獨立工作目錄 `runs/k3_seed<SEED>/`；
該 seed 底下依序跑完 sweep 的每個點。`SEED = config.seed + task id`，與本地 `main.py` 相同。

> 這些腳本只在本地驗證過 `run_seed.py` / `merge_results.py` / `progress.py`（小規模、k=1、不含 QSTA），
> `sbatch` 流程與 k=3 的實際耗時都還沒在 NCHC 上跑過。`--mem` 是估計值。

---

## TL;DR

```bash
cd /work/$USER/quantum_multicast   # 所有指令都在專案目錄下跑
mkdir -p logs                      # sbatch 之前 logs/ 就要存在，否則 Slurm 寫不了 log

sbatch nchc/smoke_test.sh          # 1. 冒煙測試（ctest，小規模）
K=3 sbatch nchc/probe.sh           # 2. 探測（ct56，K=3 才是要估的），量時間與記憶體
                                   # 3. 看 probe 結果決定 k=3 是否可行  ← 決策點
sbatch nchc/run_k.sh               # 4. 正式 100 seed（第一批 0-79），分兩批，見下
python nchc/progress.py            #    看進度
python nchc/merge_results.py --runs runs --k 3 --out merged_k3
```

---

## 檔案

| 檔案 | 用途 |
| --- | --- |
| `sweep.json` | 實驗設定：config、sweep 類型（dests/qc）、sweep 值、alpha、algos、seed 數 |
| `env.sh` | 載入 conda 環境（被各腳本 source） |
| `smoke_test.sh` | 小規模冒煙測試（ctest），確認環境與 merge 都正確 |
| `probe.sh` | 1 seed、k 由 `K` 指定（預設 2）、最大 sweep 值，量 wall time 與 peak RSS |
| `run_k.sh` | 100 seed 的 job array，k 由環境變數 `K` 指定（預設 3） |
| `run_seed.py` | 單一 task 實際執行的程式：跑一個 seed 的整個 sweep |
| `progress.py` | 看進度；`--pending` 列出還沒跑完的 task id |
| `merge_results.py` | 合併各 seed 的原始結果，算出正確的 mean/std，輸出 CSV |
| `common.py` | 上述 python 腳本共用（sweep 計畫、目錄命名、完成判定） |

要換 sweep（例如改跑 dests）只改 `sweep.json`：

```json
{"config": "configs/synthetic_dests.json", "sweep": "dests", "values": [40,45,50,55,60], "alphas": [4], "algos": "all", "n_seeds": 100}
```

---

## 步驟

### 0. 建環境（登入節點，只做一次）

```bash
module load miniconda3
conda create -n qmulticast python=3.12 -y
conda activate qmulticast
pip install networkx matplotlib
```

環境名稱預設 `qmulticast`，要換用 `CONDA_ENV=<名稱>`（[env.sh](env.sh) 讀它）。

### 1. 上傳專案到 /work

**放 `/work/<帳號>`，不要放 home**（home 配額小，這個實驗會產生大量 json 與 log）。

```bash
echo $HOME          # /home/<帳號>   ← 容量小，不要放這裡
ls -d /work/$USER   # /work/<帳號>   ← 放這裡
```

```bash
rsync -av --exclude='.git' --exclude='__pycache__' \
      --exclude='experiment*' --exclude='img*' --exclude='output_graph*' \
      --exclude='checkpoints*' --exclude='runs' --exclude='paper_fig' \
      ./ <帳號>@t3-login.nchc.org.tw:/work/<帳號>/quantum_multicast/
```

腳本用 `$SLURM_SUBMIT_DIR`（`sbatch` 當下所在的目錄），所以放哪裡都能跑，
**關鍵是 `sbatch` 要在專案目錄下執行**。

### 2. 冒煙測試

```bash
sbatch nchc/smoke_test.sh
```

用 sweep 裡最小的值、k=1、2 個 seed、只跑便宜的 algo（不含 QSTA），並測試 merge。跑完檢查：

```bash
ls runs/smoke_seed*/raw/*.json
cat merged_smoke/all_aggregated.csv
```

### 3. 跑 probe（單 seed；要估 k=3 就用 `K=3`）

```bash
K=3 sbatch nchc/probe.sh        # K 預設 2；時間不夠可 sbatch --time=... 覆寫
```

```bash
grep -E "Elapsed \(wall clock\)|Maximum resident" logs/probe_*.err
```

| 數字 | 用途 |
| --- | --- |
| `Elapsed (wall clock)` | 單一 seed 跑「一個最大 sweep 值」的時間；整個 seed 要乘上點數。估 `run_k.sh` 的 `--time` |
| `Maximum resident set size` (KB) ÷ 1024² | 估 `--mem`，建議設 peak RSS 的 2 倍 |

### 4. 決策點 ← 不要跳過

本地實測（k=3、`|D|=10`、synthetic 150 nodes）QSTA 單次 build 約 260 秒，`|D|=15` 更久；
qc sweep 的 `|D|=50`、最多 390 nodes 的 k=3 耗時目前**沒有實測**，需用 `K=3` 的 probe 量（k=2 的結果不能外推到 k=3，兩者差距很大）。
單一 task 要跑完整個 sweep，若總時間超過 `ct56` 的 4 天上限，task 會被砍掉；
`run_seed.py` 每跑完一個 algo 就存檔，被砍後重送會從沒跑完的點續跑，不會整個 seed 重來。

看完 probe 再決定：直接送、降 seed 數、減少 sweep 值，或把 sweep 拆成多個 `sweep.json` 分開送。

### 5. 送正式實驗（分兩批）

ct56 的 QOS（`MaxSubmitPU=80`、`MaxJobsPU=25`）不允許一次送 100 個 task：

```bash
sbatch nchc/run_k.sh                       # 第一批：task 0-79（K 預設 3）
```

等第一批快跑完（佇列剩下少於 60 個）再送第二批：

```bash
squeue -u $USER -h | wc -l                 # 先看還剩幾個
sbatch --array=80-99%25 nchc/run_k.sh      # 第二批：task 80-99
```

k=1 同理，`K=1 sbatch nchc/run_k.sh`，結果在 `runs/k1_seed*/`。
注意 k=1 與 k=3 共用 80 的上限，不要同時送。

只想先試幾個 seed：

```bash
sbatch --array=0-4%5 nchc/run_k.sh
```

### 6. 合併結果

```bash
python nchc/merge_results.py --runs runs --k 3 --out merged_k3
```

| 輸出（`merged_k3/`） | 內容 |
| --- | --- |
| `all_aggregated.csv` | **所有 sweep 點的聚合結果（mean + `*_std`）合成一個 CSV** |
| `raw_all.csv` | 所有 seed 的未聚合列 |
| `qc_*_k_3_d*.csv` / `dests_*_k_3.csv` | 與 `main.py` 同欄位、同檔名，可直接放進本地 `experiment/` 畫圖 |

### 7. 傳回本地端

```bash
rsync -av <帳號>@t3-login.nchc.org.tw:/work/<帳號>/quantum_multicast/merged_k3/ ./nchc/merged_k3/
```

---

## 看進度

```bash
python nchc/progress.py                 # 全部
python nchc/progress.py --job 2105843   # 指定 JOBID
python nchc/progress.py --k 1           # 看 k=1
```

印三段：佇列狀態、完成度（進度條；跑到一半的 seed 會列出 `seed43(2/5)`）、正在跑的 task 各自做到哪（log 最後一行）。

| 選項 | 用途 |
| --- | --- |
| `--job <id>` | 只看某個 job array |
| `--n-tasks N` | 總 task 數（預設 `sweep.json` 的 `n_seeds`） |
| `--pending` | 只印待跑的 task id，給 `sbatch --array` 用 |
| `--k K` / `--prefix P` / `--runs DIR` | 對應 k、工作目錄前綴（預設 `k<K>`）、根目錄（預設 `runs`） |

```bash
squeue -u $USER                      # R=跑中 PD=排隊
tail -n 30 logs/k_<JOBID>_0.out      # 看某個 task 的輸出
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ReqMem
scancel <JOBID>                      # 取消整個 array
scancel <JOBID>_3                    # 只取消第 3 個 task
```

---

## 補送失敗的 seed

job 被砍掉（超時、節點故障）時，只送沒跑完的：

```bash
python nchc/progress.py --pending                          # 先看清單
sbatch --array=$(python nchc/progress.py --pending)%25 nchc/run_k.sh
```

判斷依據是 `runs/k3_seed<SEED>/raw/*.json`：sweep 的每個點、每個 algo 都嘗試過才算完成，
壞掉的 JSON 當未完成。清單若超過 80 個，要像正式實驗一樣分批。

> **不要用 `sbatch nchc/run_k.sh` 重送整批。** 那會建立另一個全新的 array，等於整批重跑
> （雖然已完成的點會很快跳過，但仍佔用佇列名額與額度）。

---

## 參考

### QOS 上限 / 佇列

| 佇列 | TIMELIMIT | MaxSubmitPU | MaxJobsPU |
| --- | --- | --- | --- |
| `ctest` | 2:00:00 | 6 | 2 |
| **`ct56`** | **4-00:00:00** | **80** | **25** |

單執行緒單節點作業用 `ct56` 最合適（時間上限最長、資源需求最小）。probe 也占 ct56 名額。
不要在登入節點直接 `python nchc/run_seed.py`，重運算一律 `sbatch`；`progress.py` 只讀檔與 `squeue`，可以在登入節點跑。

### 併發數 `%25` 與 `--mem`

`%25` 對齊 `MaxJobsPU`。同時佔用記憶體 = 併發數 × `--mem`；拿到 probe 的 peak RSS 後可調小 `--mem`。
**不要不加 `%`**，那代表無上限。

---

## 設計說明

### 每個 task 一個獨立目錄

多個 process 寫同一份檔案會互相覆蓋。每個 seed 只寫自己的 `runs/k<K>_seed<SEED>/raw/`，
暫存檔名含 pid，彼此不會衝突。腳本直接從專案根目錄 import 程式碼，不需要像其他專案那樣把 `*.py` symlink 進工作目錄
（本專案 import 時不會寫檔）。

### 合併不能取「平均的平均」

每個 task 只有 1 個 seed，task 自己的 mean 就是那一個值、std 是 0。
`merge_results.py` 讀各 seed 的**原始值**，收齊後才呼叫 `main.aggregate_runs` 聚合一次，
所以 mean/std 與本地 `main.py` 跑 100 runs 的算法一致。

### 可重現性

`env.sh` 設 `PYTHONHASHSEED=0`，固定 set 的迭代順序；本地 `main.py` 沒設，所以與本地的數字不保證逐位一致。
