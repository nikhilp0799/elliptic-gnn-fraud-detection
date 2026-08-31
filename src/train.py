"""Train and compare: XGBoost, MLP, GraphSAGE, GAT — same temporal split.

Usage:
    python -m src.train --model sage --features all
    python -m src.train --model xgb  --features local
"""

import os

# torch, xgboost and scikit-learn each bundle their own libomp.dylib on macOS.
# Having more than one OpenMP runtime loaded in the same process corrupts
# their shared thread-pool barrier state and segfaults inside XGBoost's
# hist tree method (and, less reliably, inside torch tensor ops) on Apple
# Silicon. Must be set before any of those libraries are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import class_balance, load_graph, temporal_split
from .evaluate import (calibration_audit, classification_metrics,
                       per_step_metrics, predict)
from .models import build

RESULTS = Path("results")


def train_gnn(model, data, split, epochs=200, lr=1e-3, weight_decay=5e-4,
              seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    # Illicit nodes are ~2% of labels; without weighting the model collapses
    # to predicting the majority class and reports a meaningless 98% accuracy.
    n_pos = int((data.y[split.train] == 1).sum())
    n_neg = int((data.y[split.train] == 0).sum())
    w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[split.train], data.y[split.train], weight=w)
        loss.backward()
        opt.step()
        if epoch % 25 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
    return model


def train_xgb(data, split, time_steps):
    from xgboost import XGBClassifier

    X = data.x.numpy()
    y = data.y.numpy()
    tr, te = split.train.numpy(), split.test.numpy()

    n_pos = int((y[tr] == 1).sum())
    n_neg = int((y[tr] == 0).sum())
    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="aucpr", tree_method="hist",
    )
    clf.fit(X[tr], y[tr])
    return y[te], clf.predict_proba(X[te])[:, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sage",
                   choices=["sage", "gat", "mlp", "xgb"])
    p.add_argument("--features", default="all", choices=["all", "local"])
    p.add_argument("--cut", type=int, default=34)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data, time_steps = load_graph(args.features)
    split = temporal_split(time_steps, data.labelled, cut=args.cut)

    print("train:", class_balance(data.y, split.train))
    print("test: ", class_balance(data.y, split.test))

    if args.model == "xgb":
        y_true, prob = train_xgb(data, split, time_steps)
    else:
        model = build(args.model, data.x.size(1))
        model = train_gnn(model, data, split, epochs=args.epochs,
                          seed=args.seed)
        y_true, prob = predict(model, data, split.test)

    report = {
        "model": args.model,
        "features": args.features,
        "split": {"train_steps": split.train_steps,
                  "test_steps": split.test_steps},
        "metrics": classification_metrics(y_true, prob),
        "calibration": calibration_audit(y_true, prob),
        "per_time_step": per_step_metrics(
            y_true, prob, time_steps[split.test.numpy()]),
    }

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.model}_{args.features}_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2))

    m = report["metrics"]
    c = report["calibration"]
    print(f"\n{args.model} / {args.features}")
    print(f"  precision (illicit) {m['precision_illicit']:.3f}")
    print(f"  recall    (illicit) {m['recall_illicit']:.3f}")
    print(f"  F1        (illicit) {m['f1_illicit']:.3f}")
    print(f"  PR-AUC              {m['pr_auc']:.3f}")
    print(f"  ECE                 {c['ece']:.3f}")
    print(f"  conf/correct corr   {c['confidence_correctness_corr']:.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
