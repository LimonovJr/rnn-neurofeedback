"""Whether a magnified positional code helps where the cue is weak.

Every representational target was read on the task it was trained on, where the cue is strong
enough and the positional code is not what limits behaviour. A function that is present but weak is
the case feedback on a neural signal is meant for. Weakening the cue makes one: the cued position is
then held less well, and a network whose positional code was magnified might hold it better. The
saved networks are read at a range of cue strengths, with target and distractor as on c1, without
retraining. All states of a network are read on the same seeded trials.

    python -m experiments.cue_strength --nets 10,11
"""

import argparse
import os

import numpy as np
import torch

from experiments.access import STATES, restore
from experiments.response import NEUTRAL, POSITIONAL
from src import behaviour, runs, tasks, trials
from src.model import CKPT

_, TARGET, DISTRACTOR = tasks.C1
CUES = (0.15, 0.25, 0.35, 0.46, 0.6)
SAVED = dict({"control": STATES["control"]}, **POSITIONAL, **NEUTRAL)


def read(model, env_fn, seed, n):
    """One state on two seeded samples: the threshold is chosen on the first, scored on the second."""
    first, second = (behaviour.roc_once(model, trials.seeded(env_fn, s), n) for s in (seed, seed + 1))
    best = int(np.argmax([p["d"] for p in first]))
    own = next(p for p in second if p["offset"] == 0.0)
    return dict(auc_select=behaviour.hull_area([(p["fa"], p["hit"]) for p in first]),
                auc=behaviour.hull_area([(p["fa"], p["hit"]) for p in second]),
                offset=first[best]["offset"], best_d=second[best]["d"], own_d=own["d"],
                hit=own["hit"], fa=own["fa"], abort=own["abort"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--n", type=int, default=3000, help="trials in each of the two seeded samples")
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/cue_strength_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        models = {"theta0": restore(net, "theta0")}
        for state, stem in SAVED.items():
            path = CKPT / "nf_c1_steps" / stem.format(net=net)
            if path.exists():
                models[state] = restore(net, "theta0")
                models[state].load_state_dict(torch.load(path, weights_only=True)["state_dict"],
                                              strict=False)
        for i, cue in enumerate(CUES):
            env_fn = tasks.sameloc(cue, TARGET, DISTRACTOR)
            for state, model in models.items():
                out = read(model, env_fn, 4000 + 10 * net + 2 * i, a.n)
                rows.append(dict(net=net, state=state, cue=cue,
                                 **{k: round(v, 4) for k, v in out.items()}))
            print(f"  [net {net} cue {cue:g}] " + "  ".join(
                f"{r['state']} {r['auc']:.3f}" for r in rows[-len(models):]), flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
