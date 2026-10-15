"""Feedback first, then practice of the read-out: whether a feedback signal builds anything a
decision read-out can use once the read-out is allowed to learn.

Every feedback arm was read through the frozen decision read-out, or through a read-out refitted on
c1, where nothing is missing from the dynamics. In a protocol the read-out downstream is not frozen
for good: practice of the task trains it. Here the recurrent weights are frozen at the end of a
feedback arm (saved by `dose.py --save`) and only the decision read-out is trained on the context
task, on block 57's schedule: a fresh read-out fitted by 2000 read-out-only supervised updates,
read, 2000 more, read again. θ₀ gets the same practice, so the comparison is feedback plus practice
against practice alone. `--head mlp` fits the nonlinear head of block 61 instead.

Every state of a network is read on the same two seeded samples: the threshold is chosen on the
first and scored on the second, on offsets wide enough for a conservative read-out to reach its
peak.

    python -m experiments.practice --net 10 --state template --head linear
"""

import argparse
import os

import numpy as np
import torch

from experiments.dose import TASKS, decision_states
from experiments.nf import adapt_decoder, refit_decoder
from src import behaviour, runs, trials
from src.model import CKPT, load, mlp_readout
from src.training import TrainConfig, train_supervised

ENV = TASKS["context"]["env"]
SAVED = CKPT / "practice_context" / "{state}_net{net}.pt"
OFFSETS = np.arange(-8.0, 12.01, 0.5)
HEAD_UNITS = 32


def restore(net, state):
    model = load(net, head="context")
    if state != "theta0":
        path = SAVED.with_name(SAVED.name.format(state=state, net=net))
        model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
    model.eval()
    return model


def train_head(model, updates):
    """Supervised training of the nonlinear head alone, with the settings of the linear refit."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.readout_head.parameters():
        p.requires_grad_(True)
    train_supervised(model, ENV, TrainConfig(batch_size=64, lr=1e-3, max_updates=updates,
                                             print_every=updates + 1, fa_penalty_weight=0.4,
                                             stop_on_no_miss=0))
    model.eval()


def read(model, net, n, tag):
    first, second = (behaviour.roc_once(model, trials.seeded(ENV, seed + net), n,
                                        behaviour.go_nogo, OFFSETS) for seed in (5000, 7000))
    best = int(np.argmax([p["d"] for p in first]))
    own = next(p for p in second if p["offset"] == 0.0)
    out = dict(best_d=second[best]["d"], offset=first[best]["offset"], own_d=own["d"],
               auc=behaviour.hull_area([(p["fa"], p["hit"]) for p in second]),
               auc_select=behaviour.hull_area([(p["fa"], p["hit"]) for p in first]),
               edge=float(first[best]["offset"] in (OFFSETS[0], OFFSETS[-1])))
    return {f"{tag}_{k}": round(float(v), 4) for k, v in out.items()}


def conjunction(model, net):
    """Cross-validated decodability of go against no-go from the decision-time state: what the
    dynamics hold, whatever reads them."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, go = decision_states(model, trials.seeded(ENV, 8000 + net), 1200)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return round(float(cross_val_score(clf, X, go, cv=5, scoring="accuracy").mean()), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--state", required=True,
                   help="theta0, or an arm of block 68 saved under checkpoints/practice_context")
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--n", type=int, default=4000, help="trials in each of the two seeded samples")
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/practice_context_{a.head}_{a.state}_net{a.net}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(71 + a.net)
    np.random.seed(71 + a.net)
    model = restore(a.net, a.state)
    row = dict(net=a.net, state=a.state, head=a.head, updates=a.updates,
               conjunction=conjunction(model, a.net), **read(model, a.net, a.n, "frozen"))
    if a.head == "mlp":
        model.readout_head = mlp_readout(HEAD_UNITS, 1000 + a.net)()
        train_head(model, a.updates)
    else:
        refit_decoder(model, ENV, a.net, a.updates)
    row.update(read(model, a.net, a.n, "refit"))
    if a.head == "mlp":
        train_head(model, a.updates)
    else:
        adapt_decoder(model, ENV, a.updates, 1e-3)
        model.eval()
    row.update(read(model, a.net, a.n, "longer"))
    runs.write(a.out, row)
    print(f"  [net {a.net} {a.state} {a.head}] best d' frozen {row['frozen_best_d']:.3f}  practised "
          f"{row['refit_best_d']:.3f}  twice the practice {row['longer_best_d']:.3f}  AUC "
          f"{row['frozen_auc']:.3f} / {row['refit_auc']:.3f} / {row['longer_auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
