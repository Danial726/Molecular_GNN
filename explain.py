import numpy as np
import torch
from torch_geometric.data import Batch

torch.set_num_threads(2)


def _single(data):
    b = Batch.from_data_list([data])
    return b


def _forward_prob(model, batch, task):
    out = model(batch)
    if task == "classification":
        out = torch.sigmoid(out)
    return out


def saliency(model, data, task):
    model.eval()
    batch = _single(data)
    batch.x = batch.x.clone().requires_grad_(True)
    out = _forward_prob(model, batch, task)
    out.sum().backward()
    g = (batch.x.grad * batch.x).abs().sum(dim=1)
    return g.detach().numpy()


def integrated_gradients(model, data, task, steps=20):
    model.eval()
    base = torch.zeros_like(data.x)
    total = torch.zeros(data.x.size(0))
    for a in np.linspace(1.0 / steps, 1.0, steps):
        batch = _single(data)
        batch.x = (base + a * (data.x - base)).clone().requires_grad_(True)
        out = _forward_prob(model, batch, task)
        out.sum().backward()
        total += (batch.x.grad * (data.x - base)).sum(dim=1).detach()
    return total.abs().numpy() / steps


def intrinsic_gate(model, data, task):
    model.eval()
    batch = _single(data)
    with torch.no_grad():
        imp = model.node_importance(batch)
    return imp.numpy()


def _masked_pred(model, data, keep_mask, task):
    batch = _single(data)
    x = batch.x.clone()
    x[~keep_mask] = 0.0
    batch.x = x
    with torch.no_grad():
        return float(_forward_prob(model, batch, task).item())


def faithfulness(model, data, importance, task, frac=0.25):
    n = data.x.size(0)
    if n < 2:
        return np.nan, np.nan
    k = max(1, int(round(n * frac)))
    order = np.argsort(importance)[::-1].copy()
    top = order[:k]

    batch = _single(data)
    with torch.no_grad():
        full = float(_forward_prob(model, batch, task).item())

    keep_wo_top = torch.ones(n, dtype=torch.bool)
    keep_wo_top[top] = False
    fid_plus = abs(full - _masked_pred(model, data, keep_wo_top, task))     # remove important -> big change (good)

    keep_only_top = torch.zeros(n, dtype=torch.bool)
    keep_only_top[top] = True
    fid_minus = abs(full - _masked_pred(model, data, keep_only_top, task))  # keep only important -> small change (good)

    return fid_plus, fid_minus


EXPLAINERS = {
    "saliency": saliency,
    "ig": integrated_gradients,
    "gate": intrinsic_gate,     # gated model only
}
