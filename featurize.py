import os

import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

ATOM_TYPES = [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]   # B C N O F Si P S Cl Br I
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _onehot(value, choices):
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def _atom_features(atom):
    feats = []
    feats += _onehot(atom.GetAtomicNum(), ATOM_TYPES)
    feats += _onehot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
    feats += _onehot(atom.GetFormalCharge(), [-1, 0, 1])
    feats += _onehot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    feats += _onehot(atom.GetHybridization(), HYBRIDIZATIONS)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats.append(atom.GetMass() * 0.01)
    return feats


def _bond_features(bond):
    feats = []
    feats += _onehot(bond.GetBondType(), BOND_TYPES)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    return feats


ATOM_DIM = len(_atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = len(_bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def mol_to_graph(smiles, y):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = _bond_features(b)
        src += [i, j]
        dst += [j, i]
        eattr += [bf, bf]
    if len(src) == 0:                       # single-atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([float(y)], dtype=torch.float))
    data.smiles = smiles
    data.n_heavy = mol.GetNumHeavyAtoms()
    return data


def build_graphs(df, cache_path):
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    graphs = []
    for smi, y in zip(df["smiles"].tolist(), df["label"].tolist()):
        g = mol_to_graph(smi, y)
        if g is not None:
            graphs.append(g)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(graphs, cache_path)
    return graphs
