"""Whether "one ROC, different operating points" holds at other target strengths than the one c1
was trained at.

Effects in human neurofeedback studies were often confined to one stimulus level near threshold.
Here the same saved networks (the start, and the end of the reward and supervised runs) are swept
at a range of target strengths, with cue and distractor as on c1, without retraining.

    python -m experiments.psychometric --nets 0,1
"""

import argparse
import os

import numpy as np
import torch

from src import behaviour, runs, tasks
from src.model import CKPT, load

CUE, _, DISTRACTOR = tasks.C1
STRENGTHS = (0.6, 0.75, 0.9, 1.05, 1.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--select-n", type=int, default=1000)
    p.add_argument("--report-n", type=int, default=4000)
    p.add_argument("--states", default="theta0,reward,control")
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/psychometric_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(37)
    np.random.seed(37)
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        for state, name in (("theta0", None), ("reward", f"nf_c1_reward_lam1_n10000_net{net}_r6_u1200.pt"),
                            ("control", f"nf_c1_control_lam1_n10000_lr0.0001_net{net}_r6_u1200.pt")):
            if state not in a.states.split(","):
                continue
            model = load(net, head="c1")
            if name:
                model.load_state_dict(torch.load(CKPT / "nf_c1_steps" / name, weights_only=True)
                                      ["state_dict"], strict=False)
            model.eval()
            for strength in STRENGTHS:
                s = behaviour.threshold_sweep(model, tasks.sameloc(CUE, strength, DISTRACTOR),
                                              behaviour.detection, a.select_n, a.report_n)
                own = next(pt for pt in s["roc"].split("|") if float(pt.split(":")[0]) == 0.0)
                hit, fa = (float(v) for v in own.split(":")[1:3])
                rows.append(dict(net=net, state=state, strength=strength, best_d=s["d"],
                                 offset=s["offset"], auc=s["auc"],
                                 own_d=round(behaviour.dprime(hit, fa), 4)))
                print(f"  [net {net} {state:<7} {strength}] best {s['d']:.3f} auc {s['auc']:.3f} "
                      f"own {rows[-1]['own_d']:.3f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
