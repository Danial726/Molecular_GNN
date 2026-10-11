REGRESSION = [
    "esol", "freesolv", "lipophilicity", "bace",
    "chembl214_ki", "chembl233_ki", "chembl2047_ec50",
]
CLASSIFICATION = ["bbbp", "clintox", "tox21_nrar"]
DATASETS = REGRESSION + CLASSIFICATION

MODELS = ["gcn", "gin", "gat", "attentivefp", "gated"]
SEEDS = [0, 1, 2]

SHIFTS = {
    "regression": ["scaffold", "size", "label"],
    "classification": ["scaffold", "size"],
}


def cells():
    out = []
    for ds in REGRESSION:
        for sh in SHIFTS["regression"]:
            out.append((ds, sh))
    for ds in CLASSIFICATION:
        for sh in SHIFTS["classification"]:
            out.append((ds, sh))
    return out
