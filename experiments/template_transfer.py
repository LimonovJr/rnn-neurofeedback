"""Whether a template taken from one network reads another: the model's version of a normative
target taken from a healthy group.

The template of the dose test is a go/no-go decoder fitted on the same network's supervised end
state, which no patient has. A target from other networks would be the realistic one. Each
network's own template and those of every other network are applied, unit by unit, to its
supervised end state and to its starting state, on decision-time states of trials not used for any
fit. Nothing is trained.

    python -m experiments.template_transfer --nets 10,11,12
"""

import argparse
import os

import numpy as np
import torch

from experiments.dose import SUPERVISED, TASKS, decision_states, fit_template
from src import runs, trials
from src.model import CKPT, load


def states(net, n):
    """Decision-time states of the start and of the supervised end, on the same held-out trials."""
    out = {}
    for state in ("theta0", "supervised"):
        model = load(net, head="context")
        if state == "supervised":
            path = CKPT / SUPERVISED.format(net=net)
            model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
        model.eval()
        out[state] = decision_states(model, trials.seeded(TASKS["context"]["env"], 600 + net), n)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", default="10,11,12,13,14,15,16,17,18,19,20,21,22,23,24")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="results/template_transfer_context.csv")
    a = p.parse_args()
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    nets = [int(n) for n in a.nets.split(",")]
    templates = {n: tuple(t.numpy() if torch.is_tensor(t) else t for t in fit_template(n))
                 for n in nets}
    rows = []
    for net in nets:
        for state, (X, go) in states(net, a.n).items():
            for source, (w, b) in templates.items():
                rows.append(dict(net=net, state=state, source=source, own=int(source == net),
                                 accuracy=round(float((((X @ w + b) > 0) == (go == 1)).mean()), 4)))
        own = [r["accuracy"] for r in rows if r["net"] == net and r["own"]]
        other = np.mean([r["accuracy"] for r in rows[-2 * len(nets):] if not r["own"]
                         and r["state"] == "supervised"])
        print(f"  [net {net}] own template: start {own[0]:.3f}, supervised end {own[1]:.3f}; "
              f"other networks' on the supervised end {other:.3f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
