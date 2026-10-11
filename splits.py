import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

N_BINS = 5


def murcko_scaffold(smiles):
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        scaf = ""
    return scaf if scaf else smiles          # fall back to the molecule itself


def _fp(smiles):
    m = Chem.MolFromSmiles(smiles)
    return _MORGAN.GetFingerprint(m)


def fingerprints(smiles_list):
    return [_fp(s) for s in smiles_list]


def distance_to_train(train_fps, query_fps):
    out = np.empty(len(query_fps), dtype=float)
    for i, q in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(q, train_fps)
        out[i] = 1.0 - (max(sims) if sims else 0.0)
    return out


def _bin_by_quantile(values, n_bins=N_BINS):
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=int)
    edges = np.linspace(0, len(values), n_bins + 1).astype(int)
    for b in range(n_bins):
        bins[order[edges[b]:edges[b + 1]]] = b
    return bins


def _finalize(df, train_idx, calib_idx, test_idx, kind, seed):
    smi = df["smiles"].tolist()
    train_fps = fingerprints([smi[i] for i in train_idx])
    test_fps = fingerprints([smi[i] for i in test_idx])
    dist = distance_to_train(train_fps, test_fps)
    bins = _bin_by_quantile(dist)
    return {
        "kind": kind,
        "seed": seed,
        "train_idx": np.asarray(train_idx),
        "calib_idx": np.asarray(calib_idx),
        "test_idx": np.asarray(test_idx),
        "test_distance": dist,
        "test_bin": bins,
    }


def _split_iid_pool(pool_idx, rng, calib_frac=0.2):
    pool = np.array(pool_idx)
    rng.shuffle(pool)
    k = int(round(len(pool) * (1 - calib_frac)))
    return pool[:k].tolist(), pool[k:].tolist()


def scaffold_split(df, seed, ood_frac=0.2):
    rng = np.random.default_rng(seed)
    scaf = df["smiles"].map(murcko_scaffold).tolist()
    groups = {}
    for i, s in enumerate(scaf):
        groups.setdefault(s, []).append(i)
    keys = list(groups.keys())
    rng.shuffle(keys)
    # rarer scaffolds (smaller groups) pushed toward the OOD/test pool
    keys.sort(key=lambda k: len(groups[k]))
    n_test = int(round(len(df) * ood_frac))
    test_idx, iid_idx, filled = [], [], 0
    for k in keys:
        if filled < n_test:
            test_idx += groups[k]
            filled += len(groups[k])
        else:
            iid_idx += groups[k]
    train_idx, calib_idx = _split_iid_pool(iid_idx, rng)
    return _finalize(df, train_idx, calib_idx, test_idx, "scaffold", seed)


def size_split(df, seed, ood_frac=0.3):
    rng = np.random.default_rng(seed)
    sizes = df["smiles"].map(lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms()).to_numpy()
    cut = np.quantile(sizes, 1 - ood_frac)
    test_idx = np.where(sizes > cut)[0].tolist()
    iid_idx = np.where(sizes <= cut)[0].tolist()
    train_idx, calib_idx = _split_iid_pool(iid_idx, rng)
    return _finalize(df, train_idx, calib_idx, test_idx, "size", seed)


def label_split(df, seed, ood_frac=0.3):
    rng = np.random.default_rng(seed)
    y = df["label"].to_numpy()
    cut = np.quantile(y, 1 - ood_frac)
    test_idx = np.where(y > cut)[0].tolist()
    iid_idx = np.where(y <= cut)[0].tolist()
    train_idx, calib_idx = _split_iid_pool(iid_idx, rng)
    return _finalize(df, train_idx, calib_idx, test_idx, "label", seed)


def make_split(df, kind, seed):
    if kind == "scaffold":
        return scaffold_split(df, seed)
    if kind == "size":
        return size_split(df, seed)
    if kind == "label":
        return label_split(df, seed)
    raise ValueError(kind)


SHIFTS_REGRESSION = ["scaffold", "size", "label"]
SHIFTS_CLASSIFICATION = ["scaffold", "size"]
