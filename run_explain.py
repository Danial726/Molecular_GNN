import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch

from . import config, explain, featurize, models

CKPT = "results/ckpts"
OUT = "results/metrics/explain"
N_MOLS = 40
EXP_SEEDS = [0]


def _stratified(bins, n):
    idx = []
    for b in sorted(set(bins.tolist())):
        pool = np.where(bins == b)[0]
        take = max(1, n // len(set(bins.tolist())))
        idx += pool[:take].tolist()
    return idx[:n]


def process(ckpt_file, graphs_cache):
    name = os.path.basename(ckpt_file)[:-3]
    out_path = os.path.join(OUT, name + ".parquet")
    if os.path.exists(out_path):
        return "skip"
    ck = torch.load(ckpt_file, weights_only=False)
    ds, task = ck["dataset"], ck["task_type"]
    if ds not in graphs_cache:
        df = pd.read_parquet(f"data/processed/{ds}.parquet")
        graphs_cache[ds] = featurize.build_graphs(df, f"data/processed/{ds}_graphs.pt")
    g = graphs_cache[ds]

    model = models.build(ck["model"])
    model.load_state_dict(ck["model_state"])
    model.eval()

    bins = ck["test_bin"]
    dist = ck["test_distance"]
    sel = _stratified(bins, N_MOLS)
    exps = ["saliency", "ig"] + (["gate"] if ck["model"] == "gated" else [])

    rows = []
    for j in sel:
        d = g[ck["test_idx"][j]]
        imps = {}
        for ex in exps:
            imp = explain.EXPLAINERS[ex](model, d, task)
            imps[ex] = imp
            fp, fm = explain.faithfulness(model, d, imp, task)
            rows.append({"dataset": ds, "shift": ck["shift"], "model": ck["model"],
                         "seed": ck["seed"], "explainer": ex, "bin": int(bins[j]),
                         "distance": float(dist[j]), "fid_plus": fp, "fid_minus": fm})
        if "saliency" in imps and "ig" in imps and len(imps["saliency"]) > 2:
            from scipy.stats import spearmanr
            rho = spearmanr(imps["saliency"], imps["ig"]).correlation
            rows.append({"dataset": ds, "shift": ck["shift"], "model": ck["model"],
                         "seed": ck["seed"], "explainer": "sal_ig_agreement", "bin": int(bins[j]),
                         "distance": float(dist[j]), "fid_plus": float(rho) if rho == rho else np.nan,
                         "fid_minus": np.nan})
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return "done"


def main(budget_s):
    start = time.time()
    files = []
    for ds, sh in config.cells():
        for seed in EXP_SEEDS:
            for m in config.MODELS:
                p = os.path.join(CKPT, f"{ds}__{sh}__{m}__s{seed}.pt")
                if os.path.exists(p):
                    files.append(p)
    cache = {}
    done = 0
    for f in files:
        if time.time() - start > budget_s:
            break
        if process(f, cache) == "done":
            done += 1
            print(f"explain {done}: {os.path.basename(f)}  {time.time()-start:.0f}s", flush=True)
    print(f"explanation pass: {done} new shards", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=100000)
    main(ap.parse_args().budget)
