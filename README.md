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

*(To be filled in from `results/` once runs complete. Report the actual
numbers — including the cases where the GNN does not beat XGBoost, if that
is what happens.)*

| Model | Features | Precision | Recall | F1 | PR-AUC | ECE |
|---|---|---|---|---|---|---|
| XGBoost | all | | | | | |
| MLP | all | | | | | |
| GraphSAGE | all | | | | | |
| GAT | all | | | | | |
| GraphSAGE | local | | | | | |

## Data

Elliptic Data Set — https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Weber et al., *Anti-Money Laundering in Bitcoin*, KDD 2019.
