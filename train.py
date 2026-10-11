import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from . import models
from .splits import fingerprints, distance_to_train

torch.set_num_threads(2)

CKPT_DIR = "results/ckpts"
BATCH = 128
MAX_EPOCHS = 100
PATIENCE = 12
LR = 1e-3
WD = 1e-5
TRAIN_CAP = 2000        # cap training molecules per run to bound CPU cost


def ckpt_path(dataset, shift, model, seed):
    return os.path.join(CKPT_DIR, f"{dataset}__{shift}__{model}__s{seed}.pt")


def _subset(graphs, idx):
    return [graphs[i] for i in idx]


def _predict(model, loader, task):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            if task == "classification":
                out = torch.sigmoid(out)
            preds.append(out.cpu().numpy())
            ys.append(batch.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(ys)


def run_one(dataset, shift, model_name, seed, df, graphs, split, overwrite=False):
    path = ckpt_path(dataset, shift, model_name, seed)
    if os.path.exists(path) and not overwrite:
        return "skip"

    torch.manual_seed(seed)
    np.random.seed(seed)
    task = df["task_type"].iloc[0]

    train_idx = split["train_idx"]
    if len(train_idx) > TRAIN_CAP:
        rng = np.random.default_rng(seed)
        train_idx = np.sort(rng.choice(train_idx, TRAIN_CAP, replace=False))
    tr = _subset(graphs, train_idx)
    ca = _subset(graphs, split["calib_idx"])
    te = _subset(graphs, split["test_idx"])

    y_tr = np.array([g.y.item() for g in tr])
    if task == "regression":
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std() + 1e-8)
    else:
        y_mean, y_std = 0.0, 1.0
    # work on standardized targets during training for regression
    norm = [g.clone() for g in tr]
    if task == "regression":
        for g in norm:
            g.y = (g.y - y_mean) / y_std

    pos_weight = None
    if task == "classification":
        p = y_tr.mean()
        pos_weight = torch.tensor([(1 - p) / max(p, 1e-3)], dtype=torch.float)

    model = models.build(model_name)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    tr_loader = DataLoader(norm, batch_size=BATCH, shuffle=True)
    ca_loader = DataLoader(ca, batch_size=256)

    best_val, best_state, best_epoch, wait = np.inf, None, 0, 0
    curve = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            out = model(batch)
            if task == "regression":
                loss = F.mse_loss(out, batch.y)
            else:
                loss = F.binary_cross_entropy_with_logits(out, batch.y, pos_weight=pos_weight)
            loss.backward()
            opt.step()

        # validation on calibration set (early-stopping signal only)
        cp, cy = _predict(model, ca_loader, task)
        if task == "regression":
            val = float(np.mean(((cp * y_std + y_mean) - cy) ** 2))
        else:
            val = float(F.binary_cross_entropy(torch.tensor(cp).clamp(1e-6, 1 - 1e-6),
                                               torch.tensor(cy)).item())
        curve.append(val)
        if val < best_val - 1e-5:
            best_val, best_epoch, wait = val, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)

    ca_pred, ca_y = _predict(model, ca_loader, task)
    te_pred, te_y = _predict(model, DataLoader(te, batch_size=256), task)
    if task == "regression":
        ca_pred = ca_pred * y_std + y_mean
        te_pred = te_pred * y_std + y_mean

    # distance of calibration molecules to train (for shift diagnostics)
    tr_fps = fingerprints([df["smiles"].iloc[i] for i in split["train_idx"]])
    ca_fps = fingerprints([df["smiles"].iloc[i] for i in split["calib_idx"]])
    ca_dist = distance_to_train(tr_fps, ca_fps)

    out = {
        "dataset": dataset, "shift": shift, "model": model_name, "seed": seed,
        "task_type": task, "y_mean": y_mean, "y_std": y_std,
        "calib_pred": ca_pred, "calib_y": ca_y, "calib_distance": ca_dist,
        "test_pred": te_pred, "test_y": te_y,
        "test_distance": split["test_distance"], "test_bin": split["test_bin"],
        "train_idx": split["train_idx"], "calib_idx": split["calib_idx"], "test_idx": split["test_idx"],
        "best_epoch": best_epoch, "val_curve": np.array(curve),
        "model_state": best_state,
        "n_train": len(tr), "n_calib": len(ca), "n_test": len(te),
    }
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(out, path)
    return "done"
