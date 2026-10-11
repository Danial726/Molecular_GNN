import json
import os
import urllib.request
from dataclasses import dataclass, field

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"

_GLAM = "https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest"
_MACE = "https://raw.githubusercontent.com/molML/MoleculeACE/main/MoleculeACE/Data/benchmark_data"


@dataclass
class Source:
    name: str
    url: str
    smiles_col: str
    label_col: str
    task: str              # "regression" or "classification"
    note: str = ""
    drop_na: bool = True
    extra_cols: list = field(default_factory=list)   # carried through if present (e.g. cliff flags)


REGISTRY = [
    Source("esol", f"{_GLAM}/ESOL_delaney-processed.csv",
           "smiles", "measured log solubility in mols per litre", "regression", "aqueous solubility (logS)"),
    Source("freesolv", f"{_GLAM}/FreeSolv_SAMPL.csv",
           "smiles", "expt", "regression", "hydration free energy"),
    Source("lipophilicity", f"{_GLAM}/Lipophilicity.csv",
           "smiles", "exp", "regression", "octanol/water logD"),
    Source("bace", f"{_GLAM}/bace.csv",
           "smiles", "pIC50", "regression", "BACE-1 inhibition pIC50"),
    Source("bbbp", f"{_GLAM}/BBBP.csv",
           "smiles", "p_np", "classification", "blood-brain barrier penetration"),
    Source("clintox", f"{_GLAM}/clintox.csv",
           "smiles", "CT_TOX", "classification", "clinical trial toxicity"),
    Source("tox21_nrar", f"{_GLAM}/tox21.csv",
           "smiles", "NR-AR", "classification", "androgen receptor (Tox21 NR-AR)"),
    Source("chembl214_ki", f"{_MACE}/CHEMBL214_Ki.csv",
           "smiles", "y", "regression", "5-HT1A Ki (MoleculeACE)", extra_cols=["cliff_mol"]),
    Source("chembl233_ki", f"{_MACE}/CHEMBL233_Ki.csv",
           "smiles", "y", "regression", "mu opioid Ki (MoleculeACE)", extra_cols=["cliff_mol"]),
    Source("chembl2047_ec50", f"{_MACE}/CHEMBL2047_EC50.csv",
           "smiles", "y", "regression", "GPR119 EC50 (MoleculeACE)", extra_cols=["cliff_mol"]),
]


def _download(src):
    os.makedirs(RAW_DIR, exist_ok=True)
    dst = os.path.join(RAW_DIR, f"{src.name}.csv")
    if not os.path.exists(dst):
        req = urllib.request.Request(src.url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
            f.write(r.read())
    return dst


def _canon(smi):
    if not isinstance(smi, str) or not smi.strip():
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(m)


def standardize(src):
    raw = _download(src)
    df = pd.read_csv(raw)
    keep = {src.smiles_col: "smiles", src.label_col: "label"}
    cols = [src.smiles_col, src.label_col] + [c for c in src.extra_cols if c in df.columns]
    df = df[cols].rename(columns=keep)
    if src.drop_na:
        df = df.dropna(subset=["label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])

    canon = df["smiles"].map(_canon)
    df = df.assign(smiles=canon).dropna(subset=["smiles"])
    df = df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)

    df["task_type"] = src.task
    df["dataset"] = src.name
    return df


def build_all(force=False):
    os.makedirs(PROC_DIR, exist_ok=True)
    manifest = {}
    for src in REGISTRY:
        out = os.path.join(PROC_DIR, f"{src.name}.parquet")
        if os.path.exists(out) and not force:
            df = pd.read_parquet(out)
        else:
            df = standardize(src)
            df.to_parquet(out, index=False)
        pos = float((df["label"] > 0.5).mean()) if src.task == "classification" else None
        manifest[src.name] = {
            "task": src.task,
            "n": int(len(df)),
            "note": src.note,
            "source": src.url,
            "pos_rate": pos,
            "label_min": float(df["label"].min()),
            "label_max": float(df["label"].max()),
        }
    with open(os.path.join(PROC_DIR, "provenance.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    man = build_all()
    for k, v in man.items():
        extra = f"pos={v['pos_rate']:.2f}" if v["pos_rate"] is not None else f"[{v['label_min']:.2f},{v['label_max']:.2f}]"
        print(f"{k:18s} {v['task'][:5]:5s} n={v['n']:5d}  {extra}  {v['note']}")
