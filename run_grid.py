import argparse
import os
import time

import pandas as pd

from . import config, featurize, splits, train

PROC = "data/processed"
LOG = "logs/grid_progress.log"


def _log(msg):
    os.makedirs("logs", exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _graphs(ds):
    df = pd.read_parquet(f"{PROC}/{ds}.parquet")
    g = featurize.build_graphs(df, f"{PROC}/{ds}_graphs.pt")
    return df, g


def pending_total():
    total = done = 0
    for ds, sh in config.cells():
        for seed in config.SEEDS:
            for m in config.MODELS:
                total += 1
                if os.path.exists(train.ckpt_path(ds, sh, m, seed)):
                    done += 1
    return done, total


def main(budget_s):
    start = time.time()
    done0, total = pending_total()
    _log(f"start: {done0}/{total} runs already complete; budget {budget_s}s")
    ran = 0
    cache = {}
    for seed in config.SEEDS:                       # seed-major: full breadth early
        for ds, sh in config.cells():
            if time.time() - start > budget_s:
                _log(f"budget reached; ran {ran} this call; {done0 + ran}/{total} total")
                return
            pend = [m for m in config.MODELS if not os.path.exists(train.ckpt_path(ds, sh, m, seed))]
            if not pend:
                continue
            if ds not in cache:
                cache[ds] = _graphs(ds)
            df, g = cache[ds]
            sp = splits.make_split(df, sh, seed)
            for m in pend:
                if time.time() - start > budget_s:
                    _log(f"budget reached mid-cell; ran {ran} this call; {done0 + ran}/{total} total")
                    return
                t = time.time()
                train.run_one(ds, sh, m, seed, df, g, sp)
                ran += 1
                _log(f"[{done0+ran}/{total}] {ds}/{sh}/{m}/s{seed}  {time.time()-t:.1f}s")
    _log(f"COMPLETE: {done0+ran}/{total} runs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=540)   # seconds of work per invocation
    args = ap.parse_args()
    main(args.budget)
