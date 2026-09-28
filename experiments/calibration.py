"""Is each reward learner at the payoff-optimal threshold for its own false-alarm cost, when it is
read the way it learns?

Block 31 found the reward learner far too conservative for its payoff under the greedy, noise-off
read-out it is scored with, and block 34 found it at the payoff optimum once the recurrent noise is
left on and releases are sampled, which is how it trains. That was one cost. Here the learners
trained at false-alarm costs 1, 2 and 4 are swept with the noise on and sampled releases, and the
payoff-optimal offset is read under every one of the three costs: on the diagonal (the cost a
learner trained with) calibration puts it at zero, off the diagonal the sign says which way the
learner would have to move for a payoff it was not trained on.

    python -m experiments.calibration --ckpt-dir nf_c1_cost
"""

import argparse
import glob
import os

import numpy as np
import torch

from src import behaviour, runs, tasks
from src.model import CKPT, load

ENV = tasks.sameloc(*tasks.C1)
COSTS = (1.0, 2.0, 4.0)


def restore(net, path):
    model = load(net, head="c1")
    model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
    model.eval()
    return model


def at_zero(roc):
    """Hit and false-alarm rates at the decoder's own threshold, off a recorded sweep."""
    point = next(p for p in roc.split("|") if float(p.split(":")[0]) == 0.0)
    return [float(v) for v in point.split(":")[1:3]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="nf_c1_cost")
    p.add_argument("--select-n", type=int, default=2000)
    p.add_argument("--report-n", type=int, default=10000)
    p.add_argument("--nets", help="comma-separated networks to read, so a set can be split "
                                    "across processes")
    p.add_argument("--out", default="results/calibration_c1.csv")
    a = p.parse_args()
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(13)
    np.random.seed(13)
    rows = []
    nets = {int(n) for n in a.nets.split(",")} if a.nets else None
    for path in sorted(glob.glob(str(CKPT / a.ckpt_dir / "*_u1200.pt"))):
        meta = torch.load(path, weights_only=True)
        net, stem = int(meta["net"]), os.path.basename(path)[:-len("_u1200.pt")]
        if nets is not None and net not in nets:
            continue
        cost = next(float(t[3:]) for t in stem.split("_") if t.startswith("fac")) \
            if "_fac" in stem else 1.0
        model = restore(net, path)
        model.noise_at_eval = True
        for sampled in (True, False):
            s = behaviour.threshold_sweep(model, ENV, behaviour.detection, a.select_n,
                                          a.report_n, sampled=sampled)
            hit, fa = at_zero(s["roc"])
            row = dict(net=net, run=stem, trained_cost=cost,
                       readout="sampled" if sampled else "greedy", best_d=s["d"], auc=s["auc"],
                       hit_at_zero=hit, fa_at_zero=fa, roc=s["roc"],
                       **{f"payoff_c{c:g}": behaviour.payoff_optimal(s["roc"], c)[0]
                          for c in COSTS})
            rows.append(row)
            print(f"  [net {net} cost {cost:g} {row['readout']}] "
                  + "  ".join(f"c{c:g}: {row[f'payoff_c{c:g}']:+.2f}" for c in COSTS)
                  + f"  hit {hit:.3f} fa {fa:.3f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
