"""Whether the context comparison was cut short by the grid of threshold offsets.

The sweep runs from -8 to +4 on the release logit. The reward learner on the context task ends far
on the conservative side, and on most of its runs the swept curve is still rising at +4: the peak
of its ROC was never reached, the hull was closed by a straight line to (1, 1), and its area and
best-threshold d' may have been reported too low. The saved end states of both learners are swept
here on a grid that runs to +12, and the same sweep is also read cut off at +4, so the two areas
come from one draw of trials and differ only by the grid.

    python -m experiments.wide_sweep --nets 0,1
"""

import argparse
import glob
import os

import numpy as np
import torch

from src import behaviour, runs, tasks
from src.model import CKPT, load

ENV = tasks.context()
WIDE = np.arange(-8.0, 12.01, 0.5)


def hull_area(points):
    hull = []
    for point in sorted([(0.0, 0.0), (1.0, 1.0)] + points):
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (point[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (point[0] - hull[-2][0])) >= 0:
            hull.pop()
        hull.append(point)
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(hull, hull[1:]))


def states(net):
    for cond in ("reward", "control"):
        for path in sorted(glob.glob(str(CKPT / "nf_context" / f"{cond}_net{net}_r*_u8000.pt"))):
            yield cond, os.path.basename(path)[:-3], path
        path = CKPT / "nf_context_b37" / f"nf_context_{cond}_lam1_n10000_crit_net{net}_r9_u8000.pt"
        if path.exists():
            yield cond, path.name[:-3], str(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--select-n", type=int, default=2000)
    p.add_argument("--report-n", type=int, default=10000)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/wide_sweep_context_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(47)
    np.random.seed(47)
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        for cond, name, path in states(net):
            model = load(net, head="context")
            model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
            model.eval()
            s = behaviour.threshold_sweep(model, ENV, behaviour.go_nogo, a.select_n, a.report_n,
                                          coarse=WIDE)
            pts = [tuple(float(v) for v in pt.split(":")[:3]) for pt in s["roc"].split("|")]
            short = [(fa, hit) for o, hit, fa in pts if o <= 4.0]
            d = lambda rate: [behaviour.dprime(hit, fa) for o, hit, fa in pts if rate(o)]
            hits = [hit for _, hit, _ in pts]
            rows.append(dict(net=net, cond=cond, state=name, auc_wide=s["auc"],
                             auc_to_4=round(hull_area(short), 4), best_d_wide=s["d"],
                             offset_wide=s["offset"], edge_wide=s["edge"],
                             coarse_d_wide=round(max(d(lambda o: True)), 4),
                             coarse_d_to_4=round(max(d(lambda o: o <= 4.0)), 4),
                             peak_offset=float(WIDE[int(np.argmax(hits))]), roc=s["roc"]))
            r = rows[-1]
            print(f"  [net {net} {cond:<7} {name[-14:]}] AUC to +4 {r['auc_to_4']:.3f} -> wide "
                  f"{r['auc_wide']:.3f}; best d' {r['coarse_d_to_4']:.3f} -> {r['coarse_d_wide']:.3f}; "
                  f"hits peak at {r['peak_offset']:+.1f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
