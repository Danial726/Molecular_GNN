import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from . import conformal, config
from .splits import fingerprints

CKPT = "results/ckpts"
OUT = "results/metrics/core_metrics.parquet"
ALPHA = 0.1
_FP_CACHE = {}


def _np(fp_list):
    import numpy as np
    arr = np.zeros((len(fp_list), 2048), dtype=np.float32)
    from rdkit import DataStructs
    for i, fp in enumerate(fp_list):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def _fps_for(ds, idx):
    key = (ds, tuple(idx[:1]), len(idx))
    df = pd.read_parquet(f"data/processed/{ds}.parquet")
    smi = [df["smiles"].iloc[i] for i in idx]
    return _np(fingerprints(smi))


def _bin_rows(base, bins, values, metric, reducer=np.mean):
    rows = []
    for b in sorted(set(bins.tolist())):
        m = bins == b
        if m.sum() == 0:
            continue
        rows.append({**base, "bin": int(b), "metric": metric, "value": float(reducer(values[m])), "n": int(m.sum())})
    rows.append({**base, "bin": -1, "metric": metric, "value": float(reducer(values)), "n": int(len(values))})
    return rows


def _cov_rows(base, bins, lo, hi, y, width, tag):
    inside = ((y >= lo) & (y <= hi)).astype(float)
    rows = _bin_rows(base, bins, inside, f"cov_{tag}")
    rows += _bin_rows(base, bins, width, f"width_{tag}")
    return rows


def per_checkpoint(ck):
    ds, task = ck["dataset"], ck["task_type"]
    base = {"dataset": ds, "shift": ck["shift"], "model": ck["model"], "seed": ck["seed"], "task": task}
    bins = ck["test_bin"]
    dist = ck["test_distance"]
    yp, yt = ck["test_pred"], ck["test_y"]
    cp, cy, cd = ck["calib_pred"], ck["calib_y"], ck["calib_distance"]
    rows = []

    # distance per bin (x-axis anchor)
    rows += _bin_rows(base, bins, dist, "distance")

    if task == "regression":
        rows += _bin_rows(base, bins, np.abs(yp - yt), "mae")
        rows += _bin_rows(base, bins, (yp - yt) ** 2, "mse")
        # conformal
        lo, hi, w = conformal.split_conformal(cp, cy, yp, ALPHA)
        rows += _cov_rows(base, bins, lo, hi, yt, w, "split")
        lo, hi, w = conformal.mondrian_conformal(cp, cy, cd, yp, dist, ALPHA)
        rows += _cov_rows(base, bins, lo, hi, yt, w, "mondrian")
        cfp = _fps_for(ds, ck["calib_idx"]); tfp = _fps_for(ds, ck["test_idx"])
        lo, hi, w = conformal.weighted_conformal(cp, cy, cfp, yp, tfp, ALPHA)
        rows += _cov_rows(base, bins, lo, hi, yt, w, "weighted")
    else:
        pred_label = (yp >= 0.5).astype(int)
        rows += _bin_rows(base, bins, (pred_label == yt).astype(float), "acc")
        rows += _bin_rows(base, bins, (yp - yt) ** 2, "brier")
        # ECE per bin
        for b in sorted(set(bins.tolist())):
            m = bins == b
            if m.sum() >= 5:
                rows.append({**base, "bin": int(b), "metric": "ece",
                             "value": conformal.ece(yp[m], yt[m]), "n": int(m.sum())})
        rows.append({**base, "bin": -1, "metric": "ece", "value": conformal.ece(yp, yt), "n": len(yt)})
        try:
            rows.append({**base, "bin": -1, "metric": "auroc", "value": float(roc_auc_score(yt, yp)), "n": len(yt)})
        except ValueError:
            pass
        in_set, size, q = conformal.lac_sets(cp, cy, yp, ALPHA)
        covered = in_set[np.arange(len(yt)), yt.astype(int)].astype(float)
        rows += _bin_rows(base, bins, covered, "cov_lac")
        rows += _bin_rows(base, bins, size.astype(float), "setsize_lac")
    return rows


def build():
    rows = []
    files = sorted(glob.glob(os.path.join(CKPT, "*.pt")))
    for i, f in enumerate(files):
        ck = torch.load(f, weights_only=False)
        rows += per_checkpoint(ck)
        if (i + 1) % 50 == 0:
            print(f"metrics {i+1}/{len(files)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows from {len(files)} checkpoints")
    return df


if __name__ == "__main__":
    build()
