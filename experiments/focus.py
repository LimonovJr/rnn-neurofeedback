"""A representational target confined to one sector of positions, read against the rest of the disk
inside the same network.

Human decoded-neurofeedback results are convincing when the effect is specific: the trained
orientation improves and the untrained ones, in the same person, serve as the control. Here the
magnified target is applied only to trials cued inside one sector (`nf --focus-theta`); trials cued
elsewhere get no signal. Each network is swept on trials cued inside that sector and on trials cued
in the opposite one, at the start and at the end. The difference of the two changes removes
everything that differs between networks, which is most of the variance in the between-network
comparisons.

    python -m experiments.focus --nets 10,11
"""

import argparse
import glob
import math
import os

import numpy as np
import torch

from experiments.induction import head_fit
from src import behaviour, runs, signals, tasks, trials
from src.model import CKPT, load


def read(model, centre, width, n_select, n_report):
    out = {}
    for name, c in (("inside", centre), ("opposite", centre + math.pi)):
        s = behaviour.threshold_sweep(model, tasks.sector(c, width), behaviour.detection, n_select,
                                      n_report)
        own = next(pt for pt in s["roc"].split("|") if float(pt.split(":")[0]) == 0.0)
        hit, fa = (float(v) for v in own.split(":")[1:3])
        # the manipulation check: the head's gain on trials cued in this sector
        batches = [trials.batch(tasks.sector(c, width), 128, 20, window_ms=signals.WINDOW_MS)
                   for _ in range(4)]
        out.update({f"{name}_best": s["d"], f"{name}_auc": s["auc"], f"{name}_offset": s["offset"],
                    f"{name}_own": round(behaviour.dprime(hit, fa), 4),
                    f"{name}_gain": round(head_fit(model, batches)[0], 4)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--glob", default="nf_c1_expand_*_foc*_net{net}_r6_u1200.pt")
    p.add_argument("--select-n", type=int, default=1500)
    p.add_argument("--report-n", type=int, default=6000)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/focus_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(41)
    np.random.seed(41)
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        for path in sorted(glob.glob(str(CKPT / "nf_c1_steps" / a.glob.format(net=net)))):
            stem = os.path.basename(path)[:-len("_u1200.pt")]
            meta = next(r for r in runs.read(f"results/{stem}.csv"))
            centre, width = math.radians(float(meta["focus_theta"])), math.radians(float(meta["focus_width"]))
            for upd, state in ((0, CKPT / "nf_c1_steps" / f"{stem}_u0.pt"), (1200, path)):
                model = load(net, head="c1")
                model.load_state_dict(torch.load(state, weights_only=True)["state_dict"], strict=False)
                model.eval()
                rows.append(dict(net=net, run=stem, upd=upd,
                                 **read(model, centre, width, a.select_n, a.report_n)))
                r = rows[-1]
                print(f"  [net {net} u{upd}] inside best {r['inside_best']:.3f} opposite best "
                      f"{r['opposite_best']:.3f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
