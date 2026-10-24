"""看 NCHC 實驗的進度（在登入節點、專案目錄下執行；只讀檔與 squeue，不佔運算資源）。

印三段：佇列狀態、完成度（進度條）、正在跑的 task 各自做到哪（抄自 log 最後一行）。
--pending 只印未完成的 task id（逗號分隔），給 sbatch --array 用。
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess

import common


def squeue(job: str | None) -> list:
    cmd = ["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%i|%T|%j"]
    if job:
        cmd += ["-j", job]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.split("|") for line in out.splitlines() if line.strip()]


def last_line(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        return lines[-1] if lines else "(log 尚無輸出)"
    except OSError:
        return "(找不到 log)"


def compress(ids: list) -> str:
    return ",".join(str(i) for i in ids)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job", default=None, help="只看某個 JOBID")
    p.add_argument("--n-tasks", type=int, default=None, help="總 task（seed）數，預設 sweep.json 的 n_seeds")
    p.add_argument("--pending", action="store_true", help="只印待跑的 task id")
    p.add_argument("--runs", default="runs")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--prefix", default=None)
    p.add_argument("--plan", default=None)
    p.add_argument("--logs", default="logs")
    a = p.parse_args()

    plan = common.load_plan(a.plan, n_seeds=a.n_tasks)
    prefix = a.prefix or f"k{a.k}"
    n = plan["n_seeds"]

    status = [common.point_status(a.runs, prefix, plan, a.k, t) for t in range(n)]
    pending = [t for t, (d, tot) in enumerate(status) if d < tot]
    if a.pending:
        print(compress(pending))
        return

    jobs = squeue(a.job)
    print("== 佇列 ==")
    states: dict = {}
    for _, st, _ in jobs:
        states[st] = states.get(st, 0) + 1
    print("  " + (", ".join(f"{k}={v}" for k, v in sorted(states.items())) or "（沒有排隊或執行中的 task）"))

    finished = n - len(pending)
    bar = "#" * int(30 * finished / n) + "." * (30 - int(30 * finished / n))
    print(f"\n== 完成度 (k={a.k}, 每個 seed 要跑 {len(common.points(plan))} 個點) ==")
    print(f"  [{bar}] {finished}/{n} seeds")
    partial = [(t, d, tot) for t, (d, tot) in enumerate(status) if 0 < d < tot]
    if partial:
        print("  跑到一半: " + ", ".join(f"seed{common.base_seed(plan) + t}({d}/{tot})" for t, d, tot in partial[:10]))
    if pending:
        print(f"  待跑 task id: {compress(pending)}")

    print("\n== 執行中 ==")
    running = [jid for jid, st, _ in jobs if st == "RUNNING"]
    if not running:
        print("  （沒有）")
    for jid in running:
        array_id, _, task = jid.partition("_")
        matches = glob.glob(os.path.join(a.logs, f"k*_{array_id}_{task}.out"))
        print(f"  task {task}: {last_line(matches[0]) if matches else '(找不到 log)'}")


if __name__ == "__main__":
    main()
