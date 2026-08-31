# Graph Neural Networks for Illicit Transaction Detection

Node classification on the Elliptic Bitcoin graph — 203,769 transactions,
234,355 edges, 49 time steps — asking one question:

**Does the graph structure actually help, or do the node features already
contain everything?**

Most GNN portfolio projects skip that question. This one is built around it.

---

## Design

**Graph-free control.** XGBoost and an MLP consume the same node features
with the graph removed. The gap between those and the GNNs is the measured
value of topology. Without the control, a good GNN score says nothing.

**Temporal split, not random.** Training on time steps 1–34, testing on
35–49. Illicit clusters span consecutive steps, so a random split routinely
puts a test node's neighbours in training — the model then scores well by
memorising cluster membership rather than learning structure.

**Two feature modes.** The Elliptic features include 72 pre-aggregated
neighbourhood statistics, which already encode graph information. Running
with `--features local` (93 local features only) forces the GNN to earn its
result from message passing rather than from features that did the
aggregation for it.

**Reported on the illicit class.** ~2% of labelled nodes are illicit;
accuracy is meaningless. Precision, recall, F1 and PR-AUC on the positive
class, plus class-weighted loss.

**Calibration audit.** Expected Calibration Error, per-bin confidence versus
observed rate, and the correlation between confidence and correctness. A
model can rank well and still be badly calibrated — which matters if anyone
downstream routes review capacity by model confidence.

**Per-time-step breakdown.** Performance is reported for each test step, not
as a single number. The Elliptic paper documents a sharp drop after a
dark-market shutdown around step 43; a single aggregate hides it entirely.

---

## Layout

```
src/data.py       graph construction, temporal split
src/models.py     GraphSAGE, GAT, MLP control
src/evaluate.py   imbalanced metrics, calibration audit, per-step breakdown
src/train.py      training and comparison entry point
results/          one JSON report per run
```

## Running

```bash
pip install torch torch-geometric pandas scikit-learn xgboost
# place the three Elliptic CSVs in data/raw/

python -m src.train --model xgb  --features all     # graph-free baseline
python -m src.train --model mlp  --features all     # neural, no topology
python -m src.train --model sage --features all     # GraphSAGE
python -m src.train --model gat  --features all     # GAT
python -m src.train --model sage --features local   # topology must earn it
```

Each run writes a JSON report with metrics, calibration and the per-step
breakdown.

---

## Results

All numbers below are read directly from the JSON reports in `results/`
(test window: time steps 35-49; illicit class only).

| Model | Features | Precision | Recall | F1 | PR-AUC | ECE |
|---|---|---|---|---|---|---|
| XGBoost | all | 0.866 | 0.734 | 0.795 | 0.799 | 0.022 |
| MLP | all | 0.622 | 0.653 | 0.637 | 0.603 | 0.035 |
| GraphSAGE | all | 0.725 | 0.606 | 0.660 | 0.680 | 0.031 |
| GAT | all | 0.349 | 0.690 | 0.463 | 0.415 | 0.095 |
| GraphSAGE | local | 0.413 | 0.717 | 0.524 | 0.706 | 0.076 |

XGBoost, the graph-free baseline, beats every GNN by a wide margin on both
F1 and PR-AUC — topology did not help on this split, and the comparison
this project is built around comes out negative for the graph. GraphSAGE
edges out the graph-free MLP (F1 0.660 vs 0.637, PR-AUC 0.680 vs 0.603), so
message passing adds a small amount of value on top of the pre-aggregated
features, but not enough to close the gap to XGBoost. GAT is the worst
model in the comparison, underperforming even the plain MLP, so more
parameters and attention did not translate into better generalisation
across the temporal split. The `--features local` run complicates the
"features already did the work" story rather than confirming it: stripped
of the 72 pre-aggregated neighbourhood features, GraphSAGE's PR-AUC (0.706)
is actually slightly *higher* than the all-features run (0.680), even
though its F1 at the default 0.5 threshold is much lower (0.524 vs 0.660,
driven by a precision collapse to 0.413) — the model's ranking of illicit
accounts is at least as good from topology alone, it is just miscalibrated
at that threshold (also visible in its ECE, 0.076 vs 0.031). The
calibration audit shows XGBoost is both the best classifier and the best
calibrated (ECE 0.022); GAT is both the worst classifier and the worst
calibrated (ECE 0.095), and confidence tracks correctness only weakly
everywhere (correlation 0.30-0.45, never strong). Every model, without
exception, collapses after time step 43: PR-AUC falls from the 0.6-1.0
range in steps 35-42 to below 0.06 for nearly every step from 44 onward
(step 43 itself scores nearly 0 across all five runs) — a single aggregate
test score would completely hide this, and it lines up exactly with the
dark-market shutdown the Elliptic paper documents, so it reads as a real
distribution shift rather than a model-specific failure.

## Data

Elliptic Data Set — https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Weber et al., *Anti-Money Laundering in Bitcoin*, KDD 2019.
