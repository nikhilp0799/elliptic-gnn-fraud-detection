"""Evaluation for a severely imbalanced node-classification problem.

Roughly 2% of labelled Elliptic nodes are illicit, so accuracy is useless —
predicting "licit" everywhere scores ~98%. Everything here is reported on
the illicit class, plus a calibration check and a per-time-step breakdown.
"""

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray,
                           threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "precision_illicit": precision_score(y_true, pred, zero_division=0),
        "recall_illicit": recall_score(y_true, pred, zero_division=0),
        "f1_illicit": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "n": int(len(y_true)),
        "n_illicit": int(y_true.sum()),
    }


def calibration_audit(y_true: np.ndarray, prob: np.ndarray,
                      bins: int = 10) -> dict:
    """Does predicted confidence track observed error?

    Bins predictions by confidence and compares predicted probability with
    the observed positive rate in each bin. Expected Calibration Error is the
    weighted mean absolute gap. A model can rank well (high AUC) and still be
    badly calibrated — which matters if anyone downstream routes review
    capacity by model confidence.
    """
    edges = np.linspace(0, 1, bins + 1)
    rows, ece, total = [], 0.0, len(prob)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi)
        if m.sum() == 0:
            continue
        conf = float(prob[m].mean())
        obs = float(y_true[m].mean())
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                     "mean_confidence": conf, "observed_rate": obs,
                     "gap": conf - obs})
        ece += (m.sum() / total) * abs(conf - obs)

    # Correlation between confidence and correctness. If this is near zero,
    # confidence carries no information about whether the model is right.
    correct = (prob >= 0.5).astype(int) == y_true
    conf_in_pred = np.where(prob >= 0.5, prob, 1 - prob)
    corr = float(np.corrcoef(conf_in_pred, correct.astype(float))[0, 1])

    return {"ece": ece, "confidence_correctness_corr": corr, "bins": rows}


def per_step_metrics(y_true: np.ndarray, prob: np.ndarray,
                     steps: np.ndarray) -> list[dict]:
    """Does performance decay as test time steps move further from training?

    The Elliptic paper reports a sharp drop after the dark-market shutdown
    around step 43. Reporting a single test number hides that entirely.
    """
    out = []
    for s in sorted(np.unique(steps)):
        m = steps == s
        if m.sum() == 0 or y_true[m].sum() == 0:
            continue
        row = {"time_step": int(s)}
        row.update(classification_metrics(y_true[m], prob[m]))
        out.append(row)
    return out


@torch.no_grad()
def predict(model, data, mask) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = model(data.x, data.edge_index)
    prob = torch.softmax(logits, dim=1)[:, 1]
    return data.y[mask].cpu().numpy(), prob[mask].cpu().numpy()
