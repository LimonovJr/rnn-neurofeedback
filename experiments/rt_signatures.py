"""What a pure criterion shift does to d', c, reaction time and its spread, against what each signal
did, on c1.

A pure criterion shift is computed exactly rather than assumed: the starting network is read at a
range of offsets on its release logit, and at each the hit and false-alarm rates, d', c and the
mean and SD of the hit reaction time are recorded. That traces one line through the space of
behavioural measures. A trained network is read at its own threshold and placed against the point
on its starting network's line with the same c; what is left over is the part of the change that a
criterion shift cannot produce. This is the comparison a study reporting d', c and RT together
could make.

    python -m experiments.rt_signatures --nets 0,1
"""

import argparse
import glob
import os

import numpy as np
import torch
from scipy.stats import norm

from src import runs, tasks, trials
from src.model import CKPT, load

ENV = tasks.sameloc(*tasks.C1)
OFFSETS = np.arange(-4.0, 4.01, 0.5)


def clamp(p):
    return min(max(float(p), 0.01), 0.99)


def read(model, n, offset=0.0):
    bias = model.b_out.detach().clone()
    with torch.no_grad():
        model.b_out[1] += offset
    tr = trials.collect(model, ENV, n_trials=n, batch_size=256)
    with torch.no_grad():
        model.b_out.copy_(bias)
    hit = sum(t["train_outcome"] == "correct" for t in tr) / len(tr)
    n_d = sum(1 for t in tr if t["has_distractors"])
    fa = sum(t["train_outcome"] == "false_alarm" for t in tr) / max(n_d, 1)
    rts = np.array([t["rt_ms"] for t in tr if t["train_outcome"] == "correct"], dtype=float)
    zh, zf = norm.ppf(clamp(hit)), norm.ppf(clamp(fa))
    return dict(hit=round(hit, 4), fa=round(fa, 4), d=round(zh - zf, 4), c=round(-(zh + zf) / 2, 4),
                rt=round(rts.mean(), 2) if len(rts) else np.nan,
                rt_sd=round(rts.std(ddof=1), 2) if len(rts) > 1 else np.nan,
                abort=round(sum(t["train_outcome"] == "abort" for t in tr) / len(tr), 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--line-n", type=int, default=3000)
    p.add_argument("--point-n", type=int, default=6000)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/rt_signatures_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(23)
    np.random.seed(23)
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        start = load(net, head="c1")
        start.eval()
        for o in OFFSETS:
            rows.append(dict(net=net, run="theta0", kind="line", offset=float(o),
                             **read(start, a.line_n, float(o))))
        print(f"  [net {net}] criterion line done", flush=True)
        for path in sorted(glob.glob(str(CKPT / "nf_c1_steps" / f"*_net{net}_r6_u1200.pt"))):
            model = load(net, head="c1")
            model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
            model.eval()
            rows.append(dict(net=net, run=os.path.basename(path)[:-len("_u1200.pt")],
                             kind="point", offset=0.0, **read(model, a.point_n)))
            r = rows[-1]
            print(f"  [net {net} {r['run'][6:40]}] d'={r['d']:.3f} c={r['c']:+.3f} "
                  f"rt={r['rt']:.0f} sd={r['rt_sd']:.0f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
