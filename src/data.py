"""
Elliptic Bitcoin dataset loading and temporal splitting.

The dataset is a directed transaction graph: 203,769 nodes, 234,355 edges,
49 time steps. Each node has 166 features (the first is the time step, the
next 93 are local transaction features, the remaining 72 are aggregated
neighbourhood features). Only ~23% of nodes carry a label:
  class 1 = illicit, class 2 = licit, 'unknown' = unlabelled.

Download: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Place the three CSVs in data/raw/.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

RAW = Path("data/raw")
FEATS = RAW / "elliptic_txs_features.csv"
EDGES = RAW / "elliptic_txs_edgelist.csv"
CLASSES = RAW / "elliptic_txs_classes.csv"

# Elliptic labels illicit=1, licit=2, unknown. We map to 1/0 and drop unknown
# from the supervised objective but KEEP the nodes in the graph — message
# passing over unlabelled neighbours is most of the point of using a GNN.
LABEL_MAP = {"1": 1, "2": 0}


@dataclass
class Split:
    """Boolean node masks for one temporal split."""

    train: torch.Tensor
    test: torch.Tensor
    train_steps: tuple[int, int]
    test_steps: tuple[int, int]


def load_graph(feature_mode: str = "all") -> tuple[Data, np.ndarray]:
    """Build a PyG Data object plus the per-node time step array.

    feature_mode:
        "all"        166 features (local + aggregated neighbourhood)
        "local"      93 local features only — use this when you want the GNN
                     to earn its keep from topology rather than from features
                     that already encode neighbourhood information.
    """
    feats = pd.read_csv(FEATS, header=None)
    feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(165)]

    classes = pd.read_csv(CLASSES)
    classes["y"] = classes["class"].map(LABEL_MAP)

    df = feats.merge(classes[["txId", "y"]], on="txId", how="left")

    # Contiguous node indexing; the edge list references raw txIds.
    idx = {tx: i for i, tx in enumerate(df["txId"].values)}

    edges = pd.read_csv(EDGES)
    src = edges["txId1"].map(idx)
    dst = edges["txId2"].map(idx)
    keep = src.notna() & dst.notna()
    edge_index = torch.tensor(
        np.vstack([src[keep].values, dst[keep].values]).astype(np.int64)
    )
    # Treat the graph as undirected for message passing.
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    if feature_mode == "local":
        cols = [f"f{i}" for i in range(93)]
    else:
        cols = [f"f{i}" for i in range(165)]

    x = torch.tensor(df[cols].values, dtype=torch.float)

    y = torch.full((len(df),), -1, dtype=torch.long)
    labelled = df["y"].notna().values
    y[labelled] = torch.tensor(df.loc[labelled, "y"].values.copy(), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.labelled = torch.tensor(labelled)
    return data, df["time_step"].values


def temporal_split(time_steps: np.ndarray, labelled: torch.Tensor,
                   cut: int = 34) -> Split:
    """Split by time, not at random.

    A random split leaks future information backwards: illicit clusters span
    consecutive time steps, so neighbours of a test node routinely appear in
    training. Splitting at a time boundary is the only honest evaluation of
    whether the model generalises to activity it has not seen.
    """
    ts = torch.tensor(time_steps)
    train = (ts <= cut) & labelled
    test = (ts > cut) & labelled
    return Split(
        train=train,
        test=test,
        train_steps=(int(ts.min()), cut),
        test_steps=(cut + 1, int(ts.max())),
    )


def class_balance(y: torch.Tensor, mask: torch.Tensor) -> dict:
    sel = y[mask]
    n = len(sel)
    pos = int((sel == 1).sum())
    return {"n": n, "illicit": pos, "illicit_rate": pos / n if n else 0.0}
