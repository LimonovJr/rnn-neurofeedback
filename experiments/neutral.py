"""What training against a representational loss costs when the loss is not about position.

The magnified positional target lowers best-threshold d'. Two controls say whether that is the
price of its content. The neutral target makes the same demand of a read-out that reads nothing
in particular (`signals.neutral_step`); its end states are swept here, and the gain it induced is
measured, so that a control that cost nothing can be told from one that changed nothing. The
scrambled change adds to the starting network the magnified target's own weight change with its
entries permuted: the same entries, and so the same size, without their arrangement.

The starting network and the magnified target's end state are swept by `experiments.access` with
the same numbers of trials.

    python -m experiments.neutral --nets 10,11
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from experiments.access import read, restore
from experiments.stepwise import ENV
from src import runs, signals, trials
from src.model import CKPT


def induced(start, end, net, n=512):
    """Gain of the neutral read-out: slope of its delay-mean output on the end state against the
    starting one, over trials, and the correlation between the two."""
    w, b, reference = signals.neutral_readout(start, ENV, net)
    x, _, _, _, _, delay_mask, _ = trials.batch(ENV, n, int(start.dt), window_ms=signals.WINDOW_MS)
    share = (delay_mask / delay_mask.sum(1, keepdim=True).clamp_min(1))[..., None]
    with torch.no_grad():
        was, now = ((F.linear(m(x, return_hidden=True)[2], w, b) * share).sum(1).flatten().numpy()
                    for m in (reference, end))
    return float(np.polyfit(was, now, 1)[0]), float(np.corrcoef(was, now)[0, 1])


def scrambled(net, seed):
    """The starting network plus the magnified target's weight change with its entries permuted
    among the connections that exist. Returned with the size of the change."""
    model, end = restore(net, "theta0"), restore(net, "k2")
    change = (end.W_rec - model.W_rec).detach()
    size = float(change.norm() / model.W_rec.norm())
    live = model.rec_mask != 0
    values = change[live]
    order = torch.randperm(len(values), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        model.W_rec[live] += values[order]
    return model, size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--kappas", default="2,3")
    p.add_argument("--ckpt-dir", default="nf_c1_steps")
    p.add_argument("--stem", default="nf_c1_neutral_lam1_n2000_k{kappa}_lr0.0001_net{net}_r6_u1200.pt")
    p.add_argument("--select-n", type=int, default=1000)
    p.add_argument("--report-n", type=int, default=4000)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/neutral_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        torch.manual_seed(71 + net)
        np.random.seed(71 + net)
        start = restore(net, "theta0")
        for kappa in a.kappas.split(","):
            path = CKPT / a.ckpt_dir / a.stem.format(kappa=kappa, net=net)
            if not path.exists():
                continue
            end = restore(net, "theta0")
            end.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
            gain, r = induced(start, end, net)
            moved = float((end.W_rec - start.W_rec).norm() / start.W_rec.norm())
            row = dict(net=net, state=f"neutral k{kappa}", moved=round(moved, 4),
                       gain=round(gain, 3), r=round(r, 3))
            row.update(read(end, a.select_n, a.report_n, "frozen"))
            rows.append(row)
            print(f"  [net {net} neutral k{kappa}] moved {moved:.4f} gain {gain:.2f} "
                  f"best d' {row['frozen_best_d']:.3f}", flush=True)
        model, moved = scrambled(net, 3000 + net)
        row = dict(net=net, state="scrambled change", moved=round(moved, 4), gain="", r="")
        row.update(read(model, a.select_n, a.report_n, "frozen"))
        rows.append(row)
        print(f"  [net {net} scrambled change] moved {moved:.4f} best d' "
              f"{row['frozen_best_d']:.3f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
