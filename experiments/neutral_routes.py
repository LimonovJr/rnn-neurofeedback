"""By what route a target that is not about position moves the criterion.

The neutral target turns the learner liberal in proportion to the gain it asks for, as the
magnified positional target does. Whether it does so the same way is read here on the saved end
states: the rates at the decoder's own threshold, and the release margin's level, spread and peak
during the hold and its peak in the response window, on one batch per network shared by its
states.

    python -m experiments.neutral_routes --nets 10,11
"""

import argparse
import os

import numpy as np
import torch

from experiments.access import restore
from experiments.neutral import scrambled
from experiments.stepwise import ENV, passage
from src import behaviour, runs, signals, trials
from src.model import CKPT

NEUTRAL = "nf_c1_neutral_lam1_n2000_k{kappa}_lr0.0001_net{net}_r6_u1200.pt"


def states(net):
    yield "theta0", restore(net, "theta0")
    yield "k2", restore(net, "k2")
    for kappa in (2, 3):
        path = CKPT / "nf_c1_steps" / NEUTRAL.format(kappa=kappa, net=net)
        if path.exists():
            model = restore(net, "theta0")
            model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
            yield f"neutral k{kappa}", model
    yield "scrambled change", scrambled(net, 3000 + net)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/neutral_routes_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        torch.manual_seed(83 + net)
        np.random.seed(83 + net)
        batch = None
        for state, model in states(net):
            if batch is None:
                batch = trials.batch(ENV, 256, int(model.dt), window_ms=signals.WINDOW_MS)
            out = behaviour.measure_outcomes(model, ENV, n=a.n)
            margin = passage(model, batch)
            rows.append(dict(net=net, state=state, hit=round(out["hit"], 4), fa=round(out["fa"], 4),
                             abort=round(out["abort"], 4), level=round(margin["level"], 3),
                             sd=round(margin["sd"], 3), hold_max=round(margin["hold_max"], 3),
                             resp_max=round(margin["resp_max"], 3)))
            print(f"  [net {net} {state:<17}] hit {out['hit']:.3f} fa {out['fa']:.3f} abort "
                  f"{out['abort']:.3f}  margin level {margin['level']:+.2f} sd {margin['sd']:.2f}",
                  flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
